"""Offline identity and no-copy layout contracts for the renamed candidate."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import test_contract as c
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ID = 'qwen-image-2-1'
GROUP = 'qwen-image-2-1'
NODES = ('generate', 'edit', 'generate-enhanced', 'edit-enhanced')


def component(root: Path, payload: bytes):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'one.bin').write_bytes(payload)
    return {'repo_id': 'fixture/only', 'revision': 'a' * 40,
            'files': {'one.bin': {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}},
            'indexes': {}}


class RenamedIdentityTests(unittest.TestCase):
    def test_manifest_and_generator_use_one_generic_image_identity(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        self.assertEqual((manifest['id'], manifest['name'], manifest['version'], manifest['source']),
                         (EXTENSION_ID, 'Qwen Image 2.1', '0.3.2', 'https://github.com/DrHepa/modly-qwen-image-2-1-extension'))
        self.assertEqual(manifest['generator_class'], 'QwenImage21Generator')
        self.assertEqual([node['id'] for node in manifest['nodes']], list(NODES))
        self.assertEqual((c.QwenImage21Generator.MODEL_ID, c.QwenImage21Generator.DISPLAY_NAME),
                         (EXTENSION_ID, 'Qwen Image 2.1'))
        self.assertNotIn('character', manifest['description'].lower())
        for node in manifest['nodes']:
            self.assertNotIn('character', node['name'].lower())
            self.assertEqual(node['weight_groups'], [GROUP])

    def test_weight_plan_and_fake_roots_remain_shared_not_copied(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        main, = manifest['weight_groups']
        self.assertEqual(main['id'], GROUP)
        self.assertEqual(len(main['model_sources']), 1)
        plans = {node['id']: node.get('model_sources', []) for node in manifest['nodes']}
        self.assertEqual((len(plans['generate']), len(plans['edit']),
                          len(plans['generate-enhanced']), len(plans['edit-enhanced'])),
                         (0, 0, 1, 1))
        self.assertNotEqual(plans['generate-enhanced'][0]['repo_id'],
                            plans['edit-enhanced'][0]['repo_id'])
        for node in manifest['nodes']:
            if node['id'].startswith('edit'):
                self.assertEqual([p['name'] for p in node['input_contract']],
                                 ['image'] + [f'image_{i}' for i in range(2, 11)])
                self.assertEqual([p['required'] for p in node['input_contract']],
                                 [True] + [False] * 9)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / EXTENSION_ID
            shared = root / '_shared' / GROUP
            image = component(shared, b'synthetic shared image')
            for node in NODES:
                private = root / node
                generator = c.QwenImage21Generator(private, root / 'outputs')
                generator.node_id = node
                generator.shared_model_dirs = {GROUP: shared}
                generator._lock = {'components': {'image': image}}
                if node.endswith('-enhanced'):
                    key = 'i2i' if node.startswith('edit') else 't2i'
                    generator._lock['components'][key] = component(
                        private / 'prompt_enhancer', key.encode())
                self.assertTrue(generator.is_downloaded())
                generator._validate_assets()
                self.assertFalse((private / 'one.bin').exists())
            self.assertNotEqual((root / 'generate-enhanced/prompt_enhancer/one.bin').read_bytes(),
                                (root / 'edit-enhanced/prompt_enhancer/one.bin').read_bytes())

    def test_readme_states_optional_kernels_and_github_source(self):
        readme = (ROOT / 'README.md').read_text().lower()
        self.assertIn('causal_conv1d', readme)
        self.assertIn('fla', readme)
        self.assertIn('optional', readme)
        self.assertIn('correct but slower', readme)
        self.assertIn('source', readme)
        self.assertIn('<models_dir>/qwen-image-2-1/', readme)

    def test_fake_png_uses_renamed_output_and_metadata_key(self):
        fixture = c.GeneratorTests()
        fixture.setUp()
        try:
            output, _ = fixture.run_fake()
            self.assertTrue(output.name.startswith('qwen-image-2-1-'))
            with Image.open(output) as image:
                self.assertIn('qwen_image_2_1', image.info)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
