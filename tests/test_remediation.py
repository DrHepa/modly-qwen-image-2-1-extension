"""Regressions for the four independently confirmed pre-install findings."""
import ast
import contextlib
import hashlib
import io
import json
import os
import threading
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import test_contract as c


def runner_send():
    # Extract the current host's real dynamic-stdout protocol function, not a
    # list-appending progress substitute. Optional source path is test-only.
    host = Path(os.environ.get('MODLY_API_DIR', str(c.ROOT.parent / 'modly' / 'api'))) / 'runner.py'
    if not host.is_file(): raise unittest.SkipTest('Actual Modly runner source is unavailable on this machine')
    fn = next(n for n in ast.parse(host.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'send')
    namespace = {'sys':sys, 'json':json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(host), 'exec'), namespace)
    return namespace['send']


class ProtocolAndCleanupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = c.GeneratorTests(); self.fixture.setUp(); self.g = self.fixture.g
    def tearDown(self): self.fixture.tearDown()

    def test_actual_runner_image_progress_stays_on_stdout(self):
        send = runner_send(); out, err = io.StringIO(), io.StringIO()
        self.g.node_id = 'generate'; self.g._model = self.fixture.fake_pipeline(); self.g._memory_mode = 'offload'
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch.object(self.g, '_validate_assets'), patch.object(self.g, '_torch_generator', return_value=None):
            self.g.generate(b'', {'prompt':'portrait','steps':3,'width':256,'height':256}, lambda p,l:send({'type':'progress','pct':p,'step':l}))
        messages = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len([m for m in messages if m['step'].startswith('Denoising step')]), 3)
        self.assertTrue(any(m['step'].startswith('Decoding') for m in messages))
        self.assertFalse(any(line.startswith('{') for line in err.getvalue().splitlines()))

    def test_actual_runner_prompt_enhancer_token_progress_stays_on_stdout(self):
        send = runner_send(); out, err = io.StringIO(), io.StringIO()
        root = self.g.model_dir / 'prompt_enhancer'; root.mkdir(parents=True)
        (root/'system_prompt.txt').write_text('test fixture system prompt')
        class Tokens:
            def __init__(self, n): self.shape = (1,n)
            def __getitem__(self, key): return []
        class Inputs(dict):
            def to(self, device): return self
        class Model:
            device = 'cuda'
            def generate(self, **kwargs):
                kwargs['stopping_criteria'][0](Tokens(131), None)
                return Tokens(131)
        processor = types.SimpleNamespace(
            apply_chat_template=lambda *a,**k:Inputs(input_ids=Tokens(3)),
            tokenizer=types.SimpleNamespace(eos_token_id=1, decode=lambda *a,**k:'reasoning</think>{"rewritten_prompt":"portrait","wh_ratio":"1:1"}'))
        module = types.ModuleType('transformers'); module.LogitsProcessorList = list
        module.StoppingCriteria = object; module.StoppingCriteriaList = list
        torch = types.SimpleNamespace(manual_seed=lambda seed:None, inference_mode=contextlib.nullcontext)
        self.g._enhancer, self.g._processor = Model(), processor
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_load_enhancer'), patch.object(self.g, '_torch',return_value=torch):
            self.g._enhance('portrait', [], self.g._parameters({'prompt':'portrait', 'enhancer_thinking':'on'}), lambda p,l:send({'type':'progress','pct':p,'step':l}), None)
        messages = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertTrue(any('128 generated tokens' in m['step'] for m in messages))
        self.assertFalse(any(line.startswith('{') for line in err.getvalue().splitlines()))

    def test_enhancer_mode_budget_and_ten_reference_template_contract(self):
        root = self.g.model_dir / 'prompt_enhancer'; root.mkdir(parents=True)
        (root/'system_prompt.txt').write_text('fixture system prompt')
        class Tokens:
            def __init__(self, n): self.shape = (1, n)
            def __getitem__(self, key): return []
        class Inputs(dict):
            def to(self, device): return self
        observed = {}
        class Model:
            device = 'cuda'
            def generate(self, **kwargs):
                observed['budget'] = kwargs['max_new_tokens']
                observed['early_stop'] = kwargs['stopping_criteria'][0](Tokens(131), None)
                return Tokens(131)
        module = types.ModuleType('transformers')
        module.LogitsProcessorList = list; module.StoppingCriteria = object
        module.StoppingCriteriaList = list
        torch = types.SimpleNamespace(manual_seed=lambda seed:None, inference_mode=contextlib.nullcontext)
        for mode, references, response, ratio in (
            ('off', [c.Image.new('RGB', (1, 1)) for _ in range(10)],
             '{"rewritten_prompt":"keep <image10>","ratio_follow":"<image10>","wh_ratio":""}', '<image10>'),
            ('on', [], 'reasoning</think>{"rewritten_prompt":"portrait","wh_ratio":"1:1"}', '1:1'),
        ):
            with self.subTest(mode=mode):
                observed.clear()
                def template(messages, **kwargs):
                    observed['thinking'] = kwargs['enable_thinking']
                    observed['images'] = len([item for item in messages[1]['content'] if item['type'] == 'image'])
                    return Inputs(input_ids=Tokens(3))
                processor = types.SimpleNamespace(
                    apply_chat_template=template,
                    tokenizer=types.SimpleNamespace(eos_token_id=1, decode=lambda *a, **k:response))
                self.g._enhancer, self.g._processor = Model(), processor
                progress = []
                with patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_load_enhancer'), patch.object(self.g, '_torch', return_value=torch):
                    result = self.g._enhance('portrait', references,
                        self.g._parameters({'prompt':'portrait', 'enhancer_thinking':mode}),
                        lambda p, label:progress.append(label), None)
                self.assertEqual(result['wh_ratio'] or result['ratio_follow'], ratio)
                self.assertEqual((observed['thinking'], observed['budget'], observed['images']),
                                 (mode == 'on', 4096, len(references)))
                self.assertEqual(observed['early_stop'], mode == 'off')
                self.assertTrue(any(f'mode={mode}' in label and 'budget=4096' in label for label in progress))

    def test_enhancer_cancel_still_stops_before_parsing_or_image_load(self):
        root = self.g.model_dir / 'prompt_enhancer'; root.mkdir(parents=True)
        (root/'system_prompt.txt').write_text('fixture system prompt')
        event = threading.Event()
        class Tokens:
            def __init__(self, size): self.shape = (1, size)
        class Inputs(dict):
            def to(self, device): return self
        class Model:
            device = 'cuda'
            def generate(self, **kwargs):
                event.set()
                kwargs['stopping_criteria'][0](Tokens(35), None)
                raise AssertionError('cancel criterion should raise')
        processor = types.SimpleNamespace(
            apply_chat_template=lambda *a, **k:Inputs(input_ids=Tokens(3)),
            tokenizer=types.SimpleNamespace(eos_token_id=1, decode=lambda *a, **k:(_ for _ in ()).throw(AssertionError('decode after cancellation'))))
        module = types.ModuleType('transformers')
        module.LogitsProcessorList = list; module.StoppingCriteria = object
        module.StoppingCriteriaList = list
        torch = types.SimpleNamespace(manual_seed=lambda seed:None, inference_mode=contextlib.nullcontext)
        self.g._enhancer, self.g._processor = Model(), processor
        with patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_load_enhancer'), patch.object(self.g, '_torch', return_value=torch), patch.object(self.g, '_load_image') as image_load:
            with self.assertRaises(c.GenerationCancelled):
                self.g._enhance('portrait', [], self.g._parameters({'prompt':'portrait'}), lambda *a:None, event)
        self.assertIsNone(self.g._enhancer); self.assertIsNone(self.g._processor)
        image_load.assert_not_called()

    def test_fast_enhancer_stops_at_first_schema_valid_object(self):
        root = self.g.model_dir / 'prompt_enhancer'; root.mkdir(parents=True)
        (root/'system_prompt.txt').write_text('fixture system prompt')
        self.g.node_id = 'edit-enhanced'
        references = [c.Image.new('RGB', (1, 1)) for _ in range(10)]
        class Tokens:
            def __init__(self, generated):
                self.generated = generated
                self.shape = (1, 3 + generated)
            def __getitem__(self, key): return list(range(self.generated))
        class Inputs(dict):
            def to(self, device): return self
        seen = {'decoded_lengths': [], 'stop_at': None}
        good = '{"rewritten_prompt":"keep <image10>","ratio_follow":"<image10>","wh_ratio":""}'
        def decode(ids, **kwargs):
            seen['decoded_lengths'].append(len(ids))
            if len(ids) < 70: return '{"rewritten_prompt":'
            if seen.get('invalid_first'):
                return '{"rewritten_prompt":"","ratio_follow":"<image10>","wh_ratio":""}\n' + good
            return good + '\nRepeated trailing text'
        class Model:
            device = 'cuda'
            def generate(self, **kwargs):
                criterion = kwargs['stopping_criteria'][0]
                for count in range(1, 129):
                    if criterion(Tokens(count), None):
                        seen['stop_at'] = count
                        return Tokens(count)
                return Tokens(128)
        processor = types.SimpleNamespace(
            apply_chat_template=lambda *a, **k:Inputs(input_ids=Tokens(0)),
            tokenizer=types.SimpleNamespace(eos_token_id=1, decode=decode))
        module = types.ModuleType('transformers')
        module.LogitsProcessorList = list; module.StoppingCriteria = object
        module.StoppingCriteriaList = list
        torch = types.SimpleNamespace(manual_seed=lambda seed:None, inference_mode=contextlib.nullcontext)
        self.g._enhancer, self.g._processor = Model(), processor
        progress = []
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_load_enhancer'), patch.object(self.g, '_torch', return_value=torch):
            result = self.g._enhance('private prompt sentinel', references,
                self.g._parameters({'prompt':'private prompt sentinel'}),
                lambda pct, label:progress.append((pct, label)), None)
        self.assertEqual(result['ratio_follow'], '<image10>')
        self.assertIsNotNone(seen['stop_at'])
        self.assertLess(seen['stop_at'], 128)
        self.assertGreaterEqual(seen['stop_at'], 70)
        self.assertEqual([pct for pct, _ in progress], sorted(pct for pct, _ in progress))
        self.assertNotIn('private prompt sentinel', out.getvalue() + err.getvalue() + str(progress))
        self.assertTrue(any(length < 70 for length in seen['decoded_lengths']))

        seen['invalid_first'], seen['stop_at'] = True, None
        self.g._enhancer, self.g._processor = Model(), processor
        with patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_load_enhancer'), patch.object(self.g, '_torch', return_value=torch):
            with self.assertRaisesRegex(ValueError, 'non-empty rewritten_prompt'):
                self.g._enhance('private prompt sentinel', references,
                    self.g._parameters({'prompt':'private prompt sentinel'}), lambda *a:None, None)
        self.assertIsNone(seen['stop_at'])


    def breaking_pipeline(self, error=None):
        pipe = self.fixture.fake_pipeline()
        def broken_hooks(): raise RuntimeError('native hook cleanup failure')
        pipe.maybe_free_model_hooks = broken_hooks
        if error:
            class BrokenCall:
                def __call__(self, **kwargs): raise error
            pipe.__class__ = type('BrokenPipeline', (BrokenCall, pipe.__class__), {})
        return pipe

    def test_unload_clears_every_resource_despite_hook_and_allocator_failures(self):
        self.g._model = self.breaking_pipeline(); self.g._enhancer = object(); self.g._processor = object()
        with patch.object(self.g, '_release_memory', side_effect=RuntimeError('allocator cleanup failure')):
            self.g.unload()
        self.assertIsNone(self.g._model); self.assertIsNone(self.g._enhancer); self.assertIsNone(self.g._processor)
        self.assertFalse(self.g.is_loaded())

    def test_native_cleanup_does_not_mask_original_generation_error(self):
        original = ValueError('original inference failure')
        pipe = self.breaking_pipeline(original)
        with self.assertRaisesRegex(ValueError, 'original inference failure'):
            self.fixture.run_fake(pipeline=pipe)
        self.assertFalse(self.g.is_loaded())

    def test_cancellation_keeps_its_exception_despite_native_cleanup_failure(self):
        event = threading.Event(); pipe = self.fixture.fake_pipeline(event)
        def broken_hooks(): raise RuntimeError('native cleanup failure')
        pipe.maybe_free_model_hooks = broken_hooks
        with self.assertRaises(c.GenerationCancelled): self.fixture.run_fake(event,pipe)
        self.assertIsNone(self.g._model)
        self.assertEqual(list(self.g.outputs_dir.glob('*.png')), [])

    def test_save_error_survives_native_cleanup_failure(self):
        with patch('PIL.Image.Image.save',side_effect=OSError('original save failure')):
            with self.assertRaisesRegex(OSError,'original save failure'):
                self.fixture.run_fake(pipeline=self.breaking_pipeline())
        self.assertIsNone(self.g._model)
        self.assertEqual(list(self.g.outputs_dir.iterdir()), [])

    def test_native_cleanup_does_not_turn_saved_success_into_an_error(self):
        output, _ = self.fixture.run_fake(pipeline=self.breaking_pipeline())
        self.assertTrue(output.is_file())
        self.assertIsNone(self.g._model)
        self.assertEqual(len(list(self.g.outputs_dir.glob('*.png'))), 1)


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        payload = b'locked'
        (self.root/'one.bin').write_bytes(payload)
        self.component = {'repo_id':'test/fixture','revision':'a'*40,'files':{'one.bin':{'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}},'indexes':{}}
    def tearDown(self): self.tmp.cleanup()

    def test_shadow_weights_configs_indexes_adapters_and_symlinks_rejected(self):
        for name in ('model.safetensors','model.safetensors.index.json','config.json','adapter_config.json','adapter_model.safetensors','diffusion_pytorch_model.safetensors','weights/pytorch_model.bin'):
            with self.subTest(name=name):
                path = self.root/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b'unverified')
                with self.assertRaisesRegex(c.AssetError, 'Unexpected|unverified'): c.AssetVerifier().verify(self.root,self.component)
                path.unlink()
                if path.parent != self.root: path.parent.rmdir()
        (self.root/'model.safetensors').symlink_to(self.root/'one.bin')
        with self.assertRaisesRegex(c.AssetError, 'symlink'): c.AssetVerifier().verify(self.root,self.component)

    def test_benign_part_sidecars_and_declared_submodel_only(self):
        (self.root/'one.bin.part').write_bytes(b'partial bookkeeping')
        (self.root/'prompt_enhancer').mkdir(); (self.root/'prompt_enhancer'/'model.safetensors').write_bytes(b'checked separately')
        self.assertTrue(c.AssetVerifier().verify(self.root,self.component, managed_subdirs=('prompt_enhancer',)))
        with self.assertRaises(c.AssetError): c.AssetVerifier().verify(self.root,self.component)
        (self.root/'one.bin.part').unlink(); (self.root/'one.bin.part').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(c.AssetError,'symlink'): c.AssetVerifier().verify(self.root,self.component,managed_subdirs=('prompt_enhancer',))

    def test_enhanced_generator_also_verifies_private_submodel(self):
        shared = self.root/'shared';shared.mkdir()
        (self.root/'one.bin').rename(shared/'one.bin')
        private = self.root/'generate-enhanced'
        pe = private/'prompt_enhancer';pe.mkdir(parents=True);(pe/'one.bin').write_bytes(b'locked')
        (pe/'model.safetensors').write_bytes(b'shadow')
        generator = c.QwenImage21Generator(private,self.root/'outputs')
        generator.node_id='generate-enhanced'
        generator.shared_model_dirs={'qwen-image-2-1':shared}
        generator._lock={'components':{'image':self.component,'t2i':self.component}}
        with self.assertRaisesRegex(RuntimeError,'Unexpected unverified checkpoint file'):
            generator._validate_assets()

    def test_repeated_readiness_check_still_rejects_new_shadow_file(self):
        verifier = c.AssetVerifier(); verifier.verify(self.root,self.component)
        (self.root/'model.safetensors').write_bytes(b'unverified')
        with self.assertRaises(c.AssetError): verifier.verify(self.root,self.component)


class VenvOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'extension';self.root.mkdir()
        (self.root/'requirements.txt').write_text('fixture; never installed')
        self.payload={'python_exe':sys.executable,'ext_dir':str(self.root),'gpu_sm':121,'cuda_version':130}
    def tearDown(self):self.tmp.cleanup()
    def test_symlinked_root_rejected_before_any_installer_command(self):
        external=Path(self.tmp.name)/'external';(external/'bin').mkdir(parents=True)
        (external/'bin'/'python').symlink_to(sys.executable)
        (self.root/'venv').symlink_to(external,target_is_directory=True)
        with patch.object(c.setup,'ROOT',self.root),patch.object(c.setup,'probe_gpu',return_value=(121,130)),patch.object(c.setup,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'symlink'):c.setup.main(['setup.py',json.dumps(self.payload)])
            run.assert_not_called()

    def test_dangling_legacy_interpreter_is_preserved_before_cp312_creation(self):
        venv=self.root/'venv';(venv/'bin').mkdir(parents=True)
        (venv/'pyvenv.cfg').write_text('home = /removed/python3.11\ninclude-system-site-packages = false\n')
        (venv/'bin'/'python').symlink_to('/removed/python3.11')
        weights=self.root/'weights.keep';weights.write_bytes(b'untouched')
        selected={**c.setup.inspect_python(sys.executable),'version':[3,12]}
        owned={**selected,'prefix':str(venv),'base_prefix':selected['base_prefix']}
        commands=[]
        def inspect(path):return owned if Path(path)==venv/'bin'/'python' else selected
        def run(args,env):
            commands.append(args)
            if args[1:3]==['-m','venv']:
                (venv/'bin').mkdir(parents=True)
                (venv/'pyvenv.cfg').write_text(f'home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n')
                (venv/'bin'/'python').symlink_to(sys.executable)
        with patch.object(c.setup,'ROOT',self.root),patch.object(c.setup,'probe_gpu',return_value=(121,130)), \
             patch.object(c.setup,'inspect_python',side_effect=inspect),patch.object(c.setup,'run',side_effect=run):
            self.assertEqual(c.setup.main(['setup.py',json.dumps(self.payload)]),0)
        backups=list(self.root.glob('venv.incompatible-*'))
        self.assertEqual(len(backups),1)
        self.assertTrue((backups[0]/'bin'/'python').is_symlink())
        self.assertFalse((backups[0]/'bin'/'python').exists())
        self.assertEqual(weights.read_bytes(),b'untouched')
        self.assertEqual(commands[0][1:3],['-m','venv'])
        self.assertTrue(all(str(backups[0]) not in str(command) for command in commands))

    def test_symlinked_config_is_not_preserved_or_replaced(self):
        venv=self.root/'venv';(venv/'bin').mkdir(parents=True)
        outside=self.root/'external.cfg';outside.write_text('home = /removed/python3.11\ninclude-system-site-packages = false\n')
        (venv/'pyvenv.cfg').symlink_to(outside)
        (venv/'bin'/'python').symlink_to('/removed/python3.11')
        with patch.object(c.setup,'ROOT',self.root),patch.object(c.setup,'probe_gpu',return_value=(121,130)),patch.object(c.setup,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'pyvenv.cfg'):c.setup.main(['setup.py',json.dumps(self.payload)])
            run.assert_not_called()
        self.assertTrue(venv.exists())
        self.assertEqual(list(self.root.glob('venv.incompatible-*')),[])

    def test_missing_config_and_nonvenv_prefix_rejected(self):
        venv=self.root/'venv';(venv/'bin').mkdir(parents=True);python=venv/'bin'/'python';python.symlink_to(sys.executable)
        with self.assertRaisesRegex(RuntimeError,'pyvenv.cfg'):c.setup.validate_venv(venv,python)
        (venv/'pyvenv.cfg').write_text(f'home = {Path(sys._base_executable).parent}\ninclude-system-site-packages = false\n')
        # Adding a real cfg makes the standard interpreter symlink a valid venv.
        self.assertEqual(Path(c.setup.validate_venv(venv,python)['prefix']).resolve(),venv.resolve())
        info=c.setup.inspect_python(sys.executable);info['prefix']=info['base_prefix']
        with patch.object(c.setup,'inspect_python',return_value=info):
            with self.assertRaisesRegex(RuntimeError,'prefix'):c.setup.validate_venv(venv,python)

    def test_valid_interpreter_symlink_and_idempotent_reuse(self):
        venv=self.root/'venv';(venv/'bin').mkdir(parents=True);python=venv/'bin'/'python';python.symlink_to(sys.executable)
        (venv/'pyvenv.cfg').write_text('home = /usr/bin\ninclude-system-site-packages = false\n')
        selected=c.setup.inspect_python(sys.executable)
        owned={**selected,'prefix':str(venv),'base_prefix':'/usr'}
        def inspect(path):return owned if Path(path)==python else selected
        with patch.object(c.setup,'ROOT',self.root),patch.object(c.setup,'probe_gpu',return_value=(121,130)),patch.object(c.setup,'inspect_python',side_effect=inspect),patch.object(c.setup,'run') as run:
            self.assertEqual(c.setup.main(['setup.py',json.dumps(self.payload)]),0)
            self.assertEqual(c.setup.main(['setup.py',json.dumps(self.payload)]),0)
            self.assertTrue(all(call.args[0][1:3]!=['-m','venv'] for call in run.call_args_list))

    def test_creation_is_validated_before_pip(self):
        commands=[]
        with patch.object(c.setup,'ROOT',self.root),patch.object(c.setup,'probe_gpu',return_value=(121,130)),patch.object(c.setup,'run',side_effect=lambda args,env:commands.append(args)):
            with self.assertRaises(RuntimeError):c.setup.main(['setup.py',json.dumps(self.payload)])
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0][1:3],['-m','venv'])

if __name__=='__main__':unittest.main()
