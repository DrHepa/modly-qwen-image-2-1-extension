"""Deterministic CPU contracts; these are not model-inference acceptance tests."""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# A minimal stand-in keeps these tests independent of a Modly installation.
base = types.ModuleType('services.generators.base')
class GenerationCancelled(Exception): pass
class BaseGenerator:
    def __init__(self, model_dir, outputs_dir):
        self.model_dir, self.outputs_dir, self._model = model_dir, outputs_dir, None
        self.model_id = ''
        self.shared_model_dirs = {}
    def is_loaded(self): return self._model is not None
    def _check_cancelled(self, event):
        if event is not None and event.is_set(): raise GenerationCancelled()
base.BaseGenerator, base.GenerationCancelled = BaseGenerator, GenerationCancelled
sys.modules['services.generators.base'] = base
from PIL import Image
from assets import AssetVerifier, AssetError
from generator import QwenImage21Generator, DEFAULTS, parse_enhancement, PresencePenalty
from package_extension import package_extension
import package_extension as package_module
spec = importlib.util.spec_from_file_location('extension_setup', ROOT / 'setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def png(color=(1, 2, 3, 4)):
    b = io.BytesIO(); Image.new('RGBA', (32, 32), color).save(b, format='PNG'); return b.getvalue()


class ContractTests(unittest.TestCase):
    def test_enhanced_nodes_expose_fast_default_and_explicit_thinking(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        self.assertEqual(manifest['version'], '0.3.3')
        for node in manifest['nodes']:
            params = {p['id']: p for p in node['params_schema']}
            if node['id'].endswith('-enhanced'):
                with self.subTest(node=node['id']):
                    thinking = params['enhancer_thinking']
                    self.assertEqual((thinking['type'], thinking['default']), ('select', 'off'))
                    self.assertEqual([o['value'] for o in thinking['options']], ['off', 'on'])
                    self.assertEqual(params['enhancer_max_tokens']['default'], 4096)
                    self.assertEqual(params['enhancer_max_tokens']['max'], 32768)
            else:
                self.assertNotIn('enhancer_thinking', params)

    def test_manifest_and_shared_package(self):
        m = json.loads((ROOT / 'manifest.json').read_text())
        self.assertEqual([n['id'] for n in m['nodes']], ['generate', 'edit', 'generate-enhanced', 'edit-enhanced'])
        self.assertEqual(m['author'], 'DrHepa')
        self.assertEqual(m['source'], 'https://github.com/DrHepa/modly-qwen-image-2-1-extension')
        lock = json.loads((ROOT / 'assets.lock.json').read_text())
        for n in m['nodes']:
            self.assertNotIn('hf_repo', n)
            self.assertEqual(n['weight_groups'], ['qwen-image-2-1'])
            self.assertEqual(len(n.get('model_sources', [])), 1 if 'enhanced' in n['id'] else 0)
            if n['id'].startswith('edit'):
                self.assertEqual(n['inputs'], ['image'] * 10)
                self.assertEqual(n['input_labels'], [f'Reference {i}' for i in range(1, 11)])
            for p in n['params_schema']:
                self.assertIn(p['type'], ('int', 'float', 'select', 'string'))
                self.assertEqual(p['default'], DEFAULTS[p['id']])
            for source in m['weight_groups'][0]['model_sources'] + n.get('model_sources', []):
                component = next(c for c in lock['components'].values() if c['repo_id'] == source['repo_id'])
                self.assertEqual(source['revision'], component['revision'])
                self.assertEqual(source['checks'], sorted(component['files']))
                self.assertEqual(source['include_prefixes'], source['checks'])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'modern'
            package_extension('modern', out)
            self.assertEqual(json.loads((out / 'manifest.json').read_text()), m)
            self.assertEqual((ROOT / 'generator.py').read_bytes(), (out / 'generator.py').read_bytes())
            with self.assertRaises(FileExistsError): package_extension('modern', out)

    def test_preserved_venv_is_ignored_and_never_packaged(self):
        self.assertIn('venv.incompatible-*/', (ROOT / '.gitignore').read_text().splitlines())
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / 'source', Path(d) / 'package'
            source.mkdir()
            for name in package_module.SHIP_FILES:
                (source / name).write_bytes((ROOT / name).read_bytes())
            shutil.copytree(ROOT / 'licenses', source / 'licenses')
            backup = source / 'venv.incompatible-test' / 'bin'
            backup.mkdir(parents=True)
            (backup / 'python').write_bytes(b'legacy environment fixture')
            with patch.object(package_module, 'ROOT', source):
                package_extension('modern', output)
            self.assertFalse((output / 'venv.incompatible-test').exists())
            self.assertFalse((output / 'venv').exists())

    def test_both_edit_reference_contracts_preserve_named_handles_and_labels(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        expected = [
            {'name': 'image' if i == 1 else f'image_{i}',
             'label': f'Reference {i}', 'type': 'image', 'required': i == 1}
            for i in range(1, 11)
        ]
        for node in manifest['nodes']:
            with self.subTest(node=node['id']):
                if node['id'] in ('edit', 'edit-enhanced'):
                    self.assertEqual(node.get('input_contract'), expected)
                    self.assertEqual(node['inputs'], ['image'] * 10)
                    self.assertEqual(node['input_labels'], [p['label'] for p in expected])
                else:
                    self.assertNotIn('input_contract', node)

    def test_reference_metadata_preserves_non_identity_manifest_behavior(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for node in manifest['nodes']:
            if node['id'] in ('edit', 'edit-enhanced'):
                node.pop('input_contract', None)
            if node['id'].endswith('-enhanced'):
                node['params_schema'] = [p for p in node['params_schema'] if p['id'] != 'enhancer_thinking']
                budget = next(p for p in node['params_schema'] if p['id'] == 'enhancer_max_tokens')
                budget['default'] = 24000
                budget['tooltip'] = 'Incomplete JSON or thinking output fails explicitly. Does not silently fall back to raw model text.'
        # Identity and publication metadata are tested separately. Lock node
        # behavior, parameter values, shared main and both PE source plans.
        for key in ('id', 'name', 'version', 'source', 'generator_class'):
            manifest.pop(key)
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, 'cd8e58b6f84f19ae405052df9abfcd87f19e24f67c6b7ddbab5ffd7ae0434844')

    def test_setup_current_host_python312_and_driver580(self):
        stack = setup.select_stack('Linux', 'aarch64', 121, 130, (3, 12))
        self.assertIn('cp312-cp312', stack[0])
        self.assertIn('sha256=252f237', stack[0])
        with patch.object(setup, 'capture', return_value='12.1, 580.178.04'):
            self.assertEqual(setup.probe_gpu({'gpu_sm':121, 'cuda_version':128}), (121, 130))
        with patch.object(setup, 'capture', return_value='12.1, 570.0'):
            with self.assertRaises(RuntimeError): setup.probe_gpu({'gpu_sm':121, 'cuda_version':130})

    def test_discovery_import_without_extension_on_sys_path(self):
        code = """
import importlib.util, sys, types
base = types.ModuleType('services.generators.base')
base.BaseGenerator = type('BaseGenerator', (), {})
base.GenerationCancelled = type('GenerationCancelled', (Exception,), {})
sys.modules['services.generators.base'] = base
spec = importlib.util.spec_from_file_location('extensions.qwen_image_2_1.generator', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.QwenImage21Generator.MODEL_ID == 'qwen-image-2-1'
"""
        run = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT / 'generator.py')], cwd='/tmp', text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_complete_asset_lock(self):
        lock = json.loads((ROOT / 'assets.lock.json').read_text())
        for key, count, size in [('image', 25, 33131609424), ('t2i', 13, 18839810883), ('i2i', 13, 18839819300)]:
            c = lock['components'][key]
            self.assertEqual(len(c['files']), count)
            self.assertEqual(sum(v['size'] for v in c['files'].values()), size)
            for f in c['files'].values(): self.assertRegex(f['sha256'], r'^[a-f0-9]{64}$')
            for index, shards in c['indexes'].items():
                self.assertIn(index, c['files'])
                for shard in shards: self.assertIn(shard, c['files'])
        self.assertIn('model-00001-of-00004.safetensors', lock['components']['t2i']['files'])

    def test_setup_json_legacy_and_no_weights(self):
        payload = {'python_exe': '/usr/bin/python3', 'ext_dir': str(ROOT), 'gpu_sm': 121, 'cuda_version': 130}
        self.assertEqual(setup.parse_args(['setup.py', json.dumps(payload)]), payload)
        self.assertEqual(setup.parse_args(['setup.py', '/usr/bin/python3', str(ROOT), '121', '130']), payload)
        with self.assertRaises(ValueError): setup.parse_args(['setup.py', '{}'])
        with self.assertRaises(ValueError): setup.parse_args(['setup.py', json.dumps({**payload, 'ext_dir': '/tmp/wrong'})])
        stack = setup.select_stack('Linux', 'aarch64', 121, 130, (3, 11))
        self.assertIn('sha256=630453', stack[0])
        with self.assertRaises(RuntimeError): setup.select_stack('Darwin', 'arm64', 0, 0, (3, 11))
        with self.assertRaises(RuntimeError): setup.select_stack('Linux', 'aarch64', 121, 120, (3, 11))
        source = (ROOT / 'setup.py').read_text()
        for forbidden in ['snapshot_download', 'hf_hub_download', 'from_pretrained(', 'git lfs']:
            self.assertNotIn(forbidden, source)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.data = b'not weights, test fixture only'
        (self.root / 'one.bin').write_bytes(self.data)
        self.component = {'revision': 'a' * 40, 'repo_id': 'test/fixture', 'files': {'one.bin': {'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}}, 'indexes': {}}
        self.v = AssetVerifier()
    def tearDown(self): self.tmp.cleanup()
    def test_missing_truncated_and_same_size_limit(self):
        self.v.verify(self.root, self.component)
        (self.root / 'one.bin').write_bytes(b'x')
        with self.assertRaisesRegex(AssetError, 'size'): self.v.verify(self.root, self.component)
        (self.root / 'one.bin').write_bytes(b'x' * len(self.data))
        self.assertTrue(self.v.verify(self.root, self.component))  # Equal-size drift needs an external hash check.
        (self.root / 'one.bin').unlink()
        with self.assertRaisesRegex(AssetError, 'Models'): self.v.verify(self.root, self.component)
    def test_symlink_escape(self):
        (self.root / 'one.bin').unlink()
        (self.root / 'one.bin').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(AssetError, 'symlink'): self.v.verify(self.root, self.component)
    def test_weight_payload_is_never_opened_or_hashed(self):
        logs = []
        with patch('builtins.open', side_effect=AssertionError('payload opened')), \
             patch('assets.os.open', side_effect=AssertionError('payload opened')), \
             patch.object(Path, 'open', side_effect=AssertionError('payload opened')):
            self.assertTrue(self.v.verify(self.root, self.component, log=logs.append))
        self.assertEqual(logs, [])
        self.assertFalse(hasattr(self.v, '_digest'))
    def test_index_references_must_be_locked(self):
        data = json.dumps({'weight_map': {'weight': 'unlocked.safetensors'}}).encode()
        (self.root / 'idx.json').write_bytes(data)
        self.component['files']['idx.json'] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        self.component['indexes']['idx.json'] = ['unlocked.safetensors']
        with self.assertRaisesRegex(AssetError, 'shard'): self.v.verify(self.root, self.component)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.g = QwenImage21Generator(self.root / 'edit', self.root / 'outputs')
        self.g.node_id = 'edit'
    def tearDown(self): self.tmp.cleanup()
    def test_zero_and_ten_ordered_alpha_refs(self):
        self.g.node_id = 'generate'
        self.assertEqual(self.g._references(b'host placeholder', {}), [])
        self.g.node_id = 'edit'
        paths = []
        for i in range(9):
            p = self.root / f'ref é {i}.png'; p.write_bytes(png((i + 10, 2, 3, 4))); paths.append(str(p))
        refs = self.g._references(png(), {'extra_image_paths': paths})
        self.assertEqual(len(refs), 10)
        self.assertEqual([im.getpixel((0, 0))[0] for im in refs], [1] + list(range(10, 19)))
        self.assertEqual(refs[0].mode, 'RGBA')
        with self.assertRaises(ValueError): self.g._references(png(), {'extra_image_paths': paths + [paths[0]]})
    def test_holes_relative_workspace_and_corruption(self):
        (self.root / 'space é.png').write_bytes(png())
        with patch.dict(os.environ, {'WORKSPACE_DIR': str(self.root)}):
            refs = self.g._references(png(), {'extra_image_paths': [None, '', 'space é.png']})
        self.assertEqual(len(refs), 2)
        for bad in ['not a list', [123], [{}]]:
            with self.assertRaises(ValueError): self.g._references(png(), {'extra_image_paths': bad})
        with self.assertRaises(ValueError): self.g._references(b'bad', {})
    def test_enhancer_schema_and_truncation(self):
        result = parse_enhancement('private reasoning</think>```json\n{"rewritten_prompt":"portrait", "wh_ratio":"1:1"}\n```', 0)
        self.assertEqual(result['rewritten_prompt'], 'portrait')
        for bad in ['<think>unfinished', '{"rewritten_prompt":"x"}', '</think>raw text', '</think>{"rewritten_prompt":"x", "wh_ratio":"0:1"}', '</think>{"rewritten_prompt":"x", "ratio_follow":"<image11>","wh_ratio":""}']:
            with self.assertRaises(ValueError): parse_enhancement(bad, 2)
        self.assertEqual(parse_enhancement('</think>{"rewritten_prompt":"x", "ratio_follow":"<image2>","wh_ratio":""}', 2)['ratio_follow'], '<image2>')
    def test_direct_enhancement_is_only_valid_with_thinking_off(self):
        direct = '{"rewritten_prompt":"keep <image10>","ratio_follow":"<image10>","wh_ratio":""}'
        self.assertEqual(parse_enhancement(direct, 10, thinking=False)['ratio_follow'], '<image10>')
        self.assertEqual(parse_enhancement(f'```json\n{direct}\n```', 10, thinking=False)['rewritten_prompt'], 'keep <image10>')
        with self.assertRaisesRegex(ValueError, 'thinking block'):
            parse_enhancement(direct, 10, thinking=True)
        for bad in ('not JSON', '{"rewritten_prompt":"x"}',
                    '{"rewritten_prompt":"<image11>","ratio_follow":"<image10>","wh_ratio":""}'):
            with self.assertRaises(ValueError): parse_enhancement(bad, 10, thinking=False)
        self.assertEqual(parse_enhancement('```json\n' + direct, 10, thinking=False)['ratio_follow'], '<image10>')
    def test_fast_mode_first_complete_object_wins_without_searching(self):
        good = '{"rewritten_prompt":"keep <image10>","ratio_follow":"<image10>","wh_ratio":""}'
        for text in (f' \n{good}\nRepeated prose and {{"rewritten_prompt":"other"}}',
                     f' \n```json\n{good}\n```\nMore prose',
                     f'```\n{good}\nSecond response'):
            with self.subTest(text=text[:20]):
                self.assertEqual(parse_enhancement(text, 10, thinking=False)['rewritten_prompt'], 'keep <image10>')
        for text, message in (
            ('42\n' + good, 'JSON object'),
            ('{"rewritten_prompt":"","ratio_follow":"<image10>","wh_ratio":""}\n' + good, 'non-empty rewritten_prompt'),
            ('{"rewritten_prompt":"<image11>","ratio_follow":"<image10>","wh_ratio":""}\n' + good, 'nonexistent reference image'),
            ('preamble ' + good, 'invalid or truncated JSON'),
            ('```python\n' + good, 'code fence'),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_enhancement(text, 10, thinking=False)
        with self.assertRaisesRegex(ValueError, 'invalid or truncated JSON'):
            parse_enhancement('</think>' + good + '\n' + good, 10, thinking=True)
    def test_parameter_validation(self):
        p = self.g._parameters({'prompt': 'portrait', 'width': '512', 'seed': '42'})
        self.assertEqual(p['width'], 512)
        self.assertEqual((p['enhancer_thinking'], p['enhancer_max_tokens']), ('off', 4096))
        self.assertEqual(self.g._parameters({'prompt':'portrait', 'enhancer_thinking':'on', 'enhancer_max_tokens':32768})['enhancer_thinking'], 'on')
        for bad in [{'width': 513}, {'steps': 0}, {'seed': -2}, {'prompt': ''}, {'true_cfg_scale': float('nan')}, {'use_kv_cache': 'maybe'}, {'enhancer_thinking':'maybe'}]:
            with self.assertRaises(ValueError): self.g._parameters({'prompt': 'portrait', **bad})
    def fake_pipeline(self, event=None):
        test = self
        class Pipeline:
            def __init__(self):
                self.vae = types.SimpleNamespace(encode=lambda *a, **k: None, decode=lambda *a, **k: None)
                self.encode_prompt = lambda *a, **k: None
                self.transformer = types.SimpleNamespace(forward=lambda *a, **k: None)
                self.cleaned = False
            def maybe_free_model_hooks(self): self.cleaned = True
            def __call__(self, **kwargs):
                test.calls = kwargs
                self.encode_prompt(); self.transformer.forward()
                for i in range(kwargs['num_inference_steps']):
                    if event and i == 1: event.set()
                    out = kwargs['callback_on_step_end'](self, i, i, {'latents': 'fixture'})
                    test.assertEqual(out, {'latents': 'fixture'})
                self.vae.decode()
                return types.SimpleNamespace(images=[Image.new('RGBA', (kwargs['width'], kwargs['height']))])
        return Pipeline()
    def run_fake(self, event=None, pipeline=None):
        self.g.node_id = 'generate'; self.g._model = pipeline or self.fake_pipeline(event); self.g._memory_mode = 'offload'
        progress = []
        with patch.object(self.g, '_validate_assets'), patch.object(self.g, '_torch_generator', return_value=None):
            out = self.g.generate(b'placeholder', {'prompt': 'portrait', 'steps': 3, 'width': 256, 'height': 256}, lambda n, s: progress.append(n), event)
        return out, progress
    def test_pipeline_progress_atomic_png_and_defaults(self):
        out, progress = self.run_fake()
        self.assertTrue(out.is_absolute()); self.assertEqual(out.suffix, '.png')
        with Image.open(out) as image: self.assertEqual(image.size, (256, 256))
        self.assertEqual(progress, sorted(progress)); self.assertEqual(progress[-1], 100)
        self.assertIsNone(self.calls['image'])
        self.assertEqual(self.calls['use_kv_cache'], True)
        self.assertEqual(list(self.g.outputs_dir.glob('*.tmp')), [])
    def test_cancel_never_returns_success(self):
        event = threading.Event(); pipe = self.fake_pipeline(event)
        with self.assertRaises(GenerationCancelled): self.run_fake(event, pipe)
        self.assertEqual(list(self.g.outputs_dir.glob('*.png')), [])
        self.assertTrue(pipe.cleaned)
    def test_enhancer_failure_never_runs_image_model(self):
        self.g.node_id = 'generate-enhanced'
        with patch.object(self.g, '_check_runtime'), patch.object(self.g, '_validate_assets'), patch.object(self.g, '_enhance', side_effect=ValueError('invalid JSON')), patch.object(self.g, '_load_image') as main:
            with self.assertRaisesRegex(ValueError, 'invalid JSON'): self.g.generate(b'', {'prompt': 'portrait'})
            main.assert_not_called()
    def test_ten_references_reach_pipeline_in_order(self):
        self.g._model = self.fake_pipeline(); self.g._memory_mode = 'offload'
        refs = []
        for i in range(2, 11):
            path = self.root / f'{i}.png'; path.write_bytes(png((i, 0, 0, 255))); refs.append(str(path))
        with patch.object(self.g, '_validate_assets'), patch.object(self.g, '_torch_generator', return_value=None):
            self.g.generate(png(), {'prompt':'portrait', 'extra_image_paths':refs, 'steps':2, 'width':256, 'height':256})
        self.assertEqual([im.getpixel((0,0))[0] for im in self.calls['image']], list(range(1,11)))

    def test_cancel_during_save_has_no_success_or_final_artifact(self):
        self.g.node_id = 'generate'; self.g._model = self.fake_pipeline(); self.g._memory_mode = 'offload'
        event = threading.Event()
        def progress(n, label):
            if n == 96: event.set()
        with patch.object(self.g, '_validate_assets'), patch.object(self.g, '_torch_generator', return_value=None):
            with self.assertRaises(GenerationCancelled): self.g.generate(b'', {'prompt':'portrait', 'steps':2, 'width':256, 'height':256}, progress, event)
        self.assertEqual(list(self.g.outputs_dir.iterdir()), [])

    def test_prompt_enhancer_checkpoint_loading_is_local_and_separate(self):
        self.g.node_id = 'edit-enhanced'
        model = unittest.mock.MagicMock(); model.to.return_value.eval.return_value = model
        model_factory = unittest.mock.MagicMock(); model_factory.from_pretrained.return_value = model
        processor_factory = unittest.mock.MagicMock()
        module = types.ModuleType('transformers')
        module.AutoProcessor = processor_factory; module.AutoModelForImageTextToText = model_factory
        torch = types.SimpleNamespace(bfloat16='bf16')
        with patch.dict(sys.modules, {'transformers':module}), patch.object(self.g, '_torch', return_value=torch):
            self.g._load_enhancer()
        args, kwargs = model_factory.from_pretrained.call_args
        self.assertEqual(args[0], str(self.g.model_dir / 'prompt_enhancer'))
        self.assertTrue(kwargs['local_files_only']); self.assertFalse(kwargs['trust_remote_code'])
        self.assertIsNone(self.g._model); self.assertTrue(self.g.is_loaded())

    def test_presence_penalty_is_additive_generated_tokens_only(self):
        class Unique(list):
            def numel(self): return len(self)
        class Tokens(list):
            def unique(self): return Unique(sorted(set(self)))
        class Inputs:
            shape = (1, 5)
            def __getitem__(self, key): return Tokens([4, 4, 1, 1, 2][key[1]])
        class ScoreSlice:
            def __init__(self, owner, indices): self.owner, self.indices = owner, indices
            def __isub__(self, penalty):
                for i in self.indices: self.owner.values[i] -= penalty
                return self
        class Scores:
            values = [10.0] * 5
            def __getitem__(self, key): return ScoreSlice(self, key[1])
            def __setitem__(self, key, value): pass
        scores = Scores()
        self.assertIs(PresencePenalty(1.5, 2)(Inputs(), scores), scores)
        self.assertEqual(scores.values, [10.0, 8.5, 8.5, 10.0, 10.0])

    def test_unload(self):
        self.g._model = self.fake_pipeline(); self.g._enhancer = object(); self.g._processor = object()
        with patch.object(self.g, '_release_memory'): self.g.unload()
        self.assertIsNone(self.g._model); self.assertIsNone(self.g._enhancer); self.assertFalse(self.g.is_loaded())

if __name__ == '__main__': unittest.main()
