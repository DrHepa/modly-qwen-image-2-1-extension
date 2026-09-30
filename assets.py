"""Pinned, local-only checkpoint validation. Own integration code: MIT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


class AssetError(RuntimeError):
    """Incomplete, unsafe, or unsupported checkpoint snapshot."""


class AssetVerifier:
    """Verify hashes once per unchanged file identity in this process.

    No persistent trust marker: a new runner must hash every tensor once. The
    cache includes device/inode/size/mtime/ctime and expected digest. Any changed
    metadata invalidates it. This protects accidental drift, not a hostile OS.
    """
    def __init__(self):
        self._verified = set()

    @staticmethod
    def _identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    @staticmethod
    def _path(root, name):
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or '\\' in name:
            raise AssetError(f'Unsafe asset path: {name!r}')
        # Reject even inward symlinks: host-managed files must be real files.
        current = root
        for segment in rel.parts:
            current = current / segment
            if current.is_symlink():
                raise AssetError(f'Checkpoint symlink is not allowed: {current}')
        return current

    def _digest(self, path, cancel=None):
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as stream:
            before = self._identity(os.fstat(stream.fileno()))
            h = hashlib.sha256()
            while True:
                if cancel: cancel()
                block = stream.read(8 * 1024 * 1024)
                if not block: break
                h.update(block)
            if self._identity(os.fstat(stream.fileno())) != before:
                raise AssetError(f'Checkpoint changed while hashing: {path}')
        return h.hexdigest(), before

    def _validate_closure(self, root, files, managed_subdirs):
        # The local loaders can prefer alternate weights/configs over a checked
        # index. Reject every unlisted entry instead of merely verifying a subset.
        # Current Modly uses <locked filename>.part for resumable downloads.
        # These regular sidecars cannot be selected by the model loaders.
        allowed_files = set(files) | {name + '.part' for name in files}
        allowed_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != '.'}
        managed = set(managed_subdirs)
        if any(name != 'prompt_enhancer' for name in managed):
            raise AssetError('Only the separately verified prompt_enhancer subtree may be delegated.')
        for directory, dirs, names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in list(dirs):
                path = base / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink(): raise AssetError(f'Checkpoint symlink is not allowed: {path}')
                if relative in managed:
                    dirs.remove(name)  # Caller verifies this entire component next.
                elif relative not in allowed_dirs:
                    raise AssetError(f'Unexpected unverified checkpoint directory: {path}. Move it outside this node before loading; no files were deleted.')
            for name in names:
                path = base / name
                if path.is_symlink(): raise AssetError(f'Checkpoint symlink is not allowed: {path}')
                if path.relative_to(root).as_posix() not in allowed_files or not path.is_file():
                    raise AssetError(f'Unexpected unverified checkpoint file: {path}. Move it outside this node before loading; no files were deleted.')

    def verify(self, root: Path, component: dict, log=None, cancel=None, *, managed_subdirs=()):
        root = Path(root).absolute()
        for ancestor in (root, *root.parents):
            if ancestor.is_symlink():
                raise AssetError(f'Checkpoint root has a symlink ancestor: {ancestor}')
        files = component['files']
        self._validate_closure(root, files, managed_subdirs)
        for index, (name, expected) in enumerate(files.items(), 1):
            if cancel: cancel()
            path = self._path(root, name)
            try: info = path.stat()
            except FileNotFoundError as exc:
                raise AssetError(f'Missing checkpoint file {path}. Download this node in Modly Models; setup does not download weights.') from exc
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected['size']:
                raise AssetError(f'Checkpoint size mismatch: {path} (expected {expected["size"]}, found {info.st_size}). Repair the Models download.')
            identity = self._identity(info)
            key = (str(path), identity, expected['sha256'])
            if key not in self._verified:
                if log: log(f'Validating asset {index}/{len(files)}: {name} ({info.st_size:,} bytes)')
                digest, hashed_identity = self._digest(path, cancel)
                if digest != expected['sha256']:
                    raise AssetError(f'Checkpoint version drift or corruption: {path}. Expected pinned {component["repo_id"]}@{component["revision"]}; SHA256 mismatch. Redownload the supported snapshot from Models. Stable upstream cannot pin revisions; use a modern host if its main snapshot changed.')
                if identity != hashed_identity or self._identity(path.stat()) != identity:
                    raise AssetError(f'Checkpoint changed during validation: {path}')
                self._verified.add(key)
        for index_name, expected_shards in component.get('indexes', {}).items():
            index_path = self._path(root, index_name)
            try:
                values = json.loads(index_path.read_text(encoding='utf-8'))['weight_map'].values()
                actual = {str(PurePosixPath(index_name).parent / p) for p in values}
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise AssetError(f'Invalid checkpoint shard index: {index_path}') from exc
            if actual != set(expected_shards) or not actual.issubset(files):
                raise AssetError(f'Incomplete or unsupported shard mapping: {index_path}')
        return True


def load_lock():
    return json.loads((Path(__file__).resolve().parent / 'assets.lock.json').read_text(encoding='utf-8'))
