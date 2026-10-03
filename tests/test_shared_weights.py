"""CPU-only sharing regressions; no checkpoint downloads or inference."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import test_contract as c

GROUP = 'qwen-image-2-1'


def component(root, payload=b'synthetic shared main'):
    root.mkdir(parents=True)
    (root / 'one.bin').write_bytes(payload)
    return {'repo_id': 'test/fixture', 'revision': 'a' * 40,
            'files': {'one.bin': {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}},
            'indexes': {}}


class SharedWeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shared = self.root / 'arbitrary-host-injected-root'
        self.main = component(self.shared)
        self.g = c.QwenImage21Generator(self.root / 'generate', self.root / 'outputs')
        self.g.node_id = 'generate'
        self.g._lock = {'components': {'image': self.main}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_injection_after_constructor_and_no_node_root_reconstruction(self):
        self.assertFalse(self.g.is_downloaded())
        self.g.shared_model_dirs = {GROUP: self.shared}
        self.assertTrue(self.g.is_downloaded())
        self.g._validate_assets()
        self.assertEqual(self.g.model_dir, self.root / 'generate')
        self.assertFalse(self.g.model_dir.exists())

    def test_missing_mapping_never_falls_back_to_valid_node_checkpoint(self):
        self.main = component(self.g.model_dir)
        for mapping in (None, {}, {'different-group': self.shared}):
            with self.subTest(mapping=mapping):
                self.g.shared_model_dirs = mapping
                self.assertFalse(self.g.is_downloaded())
                with patch.object(self.g, '_load_image') as image:
                    with self.assertRaisesRegex(RuntimeError, 'shared_model_dirs.*qwen-image-2-1|qwen-image-2-1.*shared_model_dirs'):
                        self.g.load()
                    image.assert_not_called()

    def test_malformed_injected_paths_fail_before_pipeline_construction(self):
        bad = ('', 'relative/path', 123, None, self.root / '..' / 'elsewhere')
        for value in bad:
            with self.subTest(value=value):
                self.g.shared_model_dirs = {GROUP: value}
                self.assertFalse(self.g.is_downloaded())
                with patch.object(self.g, '_torch') as torch:
                    with self.assertRaisesRegex(RuntimeError, 'shared_model_dirs|shared weights'):
                        self.g._load_image()
                    torch.assert_not_called()
        self.g.shared_model_dirs = [('qwen-image-2-1', self.shared)]
        self.assertFalse(self.g.is_downloaded())
        with self.assertRaises(RuntimeError): self.g._validate_assets()

    def test_symlink_root_or_ancestor_rejected_in_readiness_and_validation(self):
        link = self.root / 'linked'; link.symlink_to(self.shared, target_is_directory=True)
        for root in (link, self.root / 'linked-parent' / self.shared.name):
            if root != link:
                (self.root / 'linked-parent').symlink_to(self.root, target_is_directory=True)
            self.g.shared_model_dirs = {GROUP: root}
            self.assertFalse(self.g.is_downloaded())
            with self.assertRaisesRegex(RuntimeError, 'symlink'): self.g._validate_assets()

    def test_shared_root_resolved_late_each_time(self):
        self.g.shared_model_dirs = {GROUP: self.shared}
        self.assertTrue(self.g.is_downloaded())
        self.g.shared_model_dirs = {GROUP: self.root / 'new-empty-host-root'}
        self.assertFalse(self.g.is_downloaded())
        with self.assertRaisesRegex(RuntimeError, 'Missing checkpoint'): self.g._validate_assets()

    def test_main_pipeline_uses_injected_path_local_only(self):
        self.g.shared_model_dirs = {GROUP: self.shared}
        factory = MagicMock()
        module = types.ModuleType('diffusers'); module.QwenImage21Pipeline = factory
        with patch.dict(sys.modules, {'diffusers': module}), patch.object(self.g, '_torch', return_value=types.SimpleNamespace(bfloat16='bf16')):
            self.g._load_image()
        factory.from_pretrained.assert_called_once_with(str(self.shared), torch_dtype='bf16', local_files_only=True)
        self.assertEqual(self.g._loaded_node, 'generate')

    def test_all_nodes_use_one_main_and_distinct_private_enhancers(self):
        for node in ('generate', 'edit', 'generate-enhanced', 'edit-enhanced'):
            with self.subTest(node=node):
                g = c.QwenImage21Generator(self.root / node, self.root / 'outputs')
                g.node_id = node; g.shared_model_dirs = {GROUP: self.shared}
                g._lock = {'components': {'image': self.main}}
                if node.endswith('-enhanced'):
                    key = 'i2i' if node.startswith('edit') else 't2i'
                    g._lock['components'][key] = component(g.model_dir / 'prompt_enhancer', key.encode())
                self.assertTrue(g.is_downloaded()); g._validate_assets()
                self.assertFalse((g.model_dir / 'one.bin').exists())
        self.assertNotEqual((self.root / 'generate-enhanced/prompt_enhancer/one.bin').read_bytes(),
                            (self.root / 'edit-enhanced/prompt_enhancer/one.bin').read_bytes())

    def test_main_and_pe_closures_checked_separately_without_delegation(self):
        self.g.node_id = 'generate-enhanced'
        self.g.shared_model_dirs = {GROUP: self.shared}
        pe = self.g.model_dir / 'prompt_enhancer'
        self.g._lock['components']['t2i'] = component(pe, b'PE')
        self.g._validate_assets()
        (self.shared / 'prompt_enhancer').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected unverified checkpoint directory'): self.g._validate_assets()
        (self.shared / 'prompt_enhancer').rmdir()
        (pe / 'model.safetensors').write_bytes(b'shadow')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected unverified checkpoint file'): self.g._validate_assets()

    def test_missing_or_wrong_size_pe_never_substitutes_shared_main(self):
        self.g.node_id = 'edit-enhanced'; self.g.shared_model_dirs = {GROUP: self.shared}
        self.g._lock['components']['i2i'] = self.main
        self.assertFalse(self.g.is_downloaded())
        with self.assertRaisesRegex(RuntimeError, 'Missing checkpoint'): self.g._validate_assets()
        component(self.g.model_dir / 'prompt_enhancer', b'x')
        with self.assertRaisesRegex(RuntimeError, 'size mismatch'): self.g._validate_assets()
        (self.g.model_dir / 'prompt_enhancer' / 'one.bin').write_bytes(b'x' * self.main['files']['one.bin']['size'])
        self.g._validate_assets()  # Equal-size content drift is outside fast readiness checks.

    def test_runtime_preflight_precedes_asset_walk(self):
        self.g.shared_model_dirs = {GROUP: self.shared}
        with patch.object(self.g, '_torch', side_effect=ImportError('no torch')), \
             patch.object(self.g, '_validate_assets') as validate:
            with self.assertRaisesRegex(RuntimeError, 'Torch is unavailable'):
                self.g.load()
            validate.assert_not_called()

    def test_missing_loader_dependency_precedes_asset_walk(self):
        self.g.shared_model_dirs = {GROUP: self.shared}
        with patch.object(self.g, '_torch'), \
             patch('generator.importlib.util.find_spec', return_value=None), \
             patch.object(self.g, '_validate_assets') as validate:
            with self.assertRaisesRegex(RuntimeError, 'diffusers is unavailable'):
                self.g.load()
            validate.assert_not_called()

    def test_manifest_declares_main_once_and_exact_descriptors(self):
        manifest = json.loads((c.ROOT / 'manifest.json').read_text())
        self.assertEqual(manifest['version'], '0.3.3')
        groups = manifest['weight_groups']; self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['id'], GROUP)
        main, = groups[0]['model_sources']
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest(main), 'f87ef8038fd7211dbd570df76decae9f15bc16a812222dce2e1732d5b4077c7d')
        expected = {'generate-enhanced': '781f24f3cfdd04d18aa683a3badfcda0d3f1ae41ed4d339695de55bf76289e42',
                    'edit-enhanced': 'dde0c320c095b33a7c67ca9b14c63495e60a23e5fe72caf7cd599765cdec3368'}
        for node in manifest['nodes']:
            self.assertEqual(node['weight_groups'], [GROUP])
            if node['id'] in expected:
                pe, = node['model_sources']; self.assertEqual(digest(pe), expected[node['id']])
            else: self.assertNotIn('model_sources', node)
        self.assertEqual(hashlib.sha256((c.ROOT / 'assets.lock.json').read_bytes()).hexdigest(),
                         'd495ad1d969df4c317dc418d20c25d55eddf6428ade98105061eeb754bcd2783')

    def test_legacy_package_rejected_before_any_output_or_parent_created(self):
        output = self.root / 'absent-parent' / 'legacy'
        with self.assertRaisesRegex(ValueError, 'shared|unsupported'):
            c.package_extension('upstream-main', output)
        self.assertFalse(output.parent.exists())


if __name__ == '__main__': unittest.main()
