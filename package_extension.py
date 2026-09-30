"""Build a root-installable shared-weights package; never alter the source."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SHIP_FILES = ('manifest.json', 'generator.py', 'assets.py', 'assets.lock.json',
              'setup.py', 'requirements.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
              'README.md', 'package_extension.py', '.gitignore')


def package_extension(target: str, output: Path):
    if target != 'modern':
        raise ValueError('Unsupported target: use modern with a shared-weights-capable Modly host. '
                         'Legacy upstream-main packaging would duplicate checkpoints and is not supported.')
    output = Path(output).absolute()
    # Require a genuinely new directory; never overwrite runtime or source.
    output.mkdir(parents=True, exist_ok=False)
    for name in SHIP_FILES:
        shutil.copy2(ROOT / name, output / name)
    shutil.copytree(ROOT / 'licenses', output / 'licenses')
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'PACKAGE_TARGET.txt').write_text(target + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=('modern',))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(package_extension(args.target, args.output))
