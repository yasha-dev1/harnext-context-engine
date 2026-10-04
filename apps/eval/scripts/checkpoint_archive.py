"""Pack, verify, or restore the frozen E1 checkpoint. Standard library only."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
GROUPS = {
    'e1-run4': 'apps/eval/out/20260907T000315Z-e1-kafka',
    'e1-run5b': 'apps/eval/out/20261004T115316Z-e1-kafka-jev-only',
    'e1-jev-decisions': 'apps/eval/out/e1-jev',
    'e2-stores-w2026': 'apps/eval/out/kafka-500-v2/stores-w2026',
    'e2-stream-w2026': 'apps/eval/out/kafka-500-v2/stream-w2026',
    'e1-kafka-corpus': 'apps/eval/out/corpus/kafka',
}

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def file_digest(p):
    with p.open('rb') as f:
        return digest(f)

def walk_files(path, ancestors=frozenset()):
    real = path.resolve()
    if path.is_dir():
        if real in ancestors:
            raise ValueError(f'Directory link cycle: {path}')
        for child in sorted(path.iterdir()):
            if child.name != '__pycache__':
                yield from walk_files(child, ancestors | {real})
    elif path.is_file():
        yield path
    else:
        raise ValueError(f'Unsupported or dangling path: {path}')

def group_files(group):
    base = ROOT / GROUPS[group]
    if group != 'e2-stores-w2026':
        yield from walk_files(base)
        return
    for store in sorted(base.iterdir()):
        if store.name != 'S-tools':
            yield from walk_files(store)
            continue
        # This alias points at the older multi-year store. Preserve the five
        # snapshots actually used in w2026, not unrelated monthly snapshots.
        yield store / 'build-state.json'
        for month in ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']:
            yield from walk_files(store / 'export' / month)
            yield store / 'export' / (month + '.manifest.json')

def pack(group, directory):
    directory.mkdir(parents=True, exist_ok=True)
    asset = directory / (group + '.tar.gz')
    if asset.exists():
        raise SystemExit(f'Refusing to overwrite {asset}')
    files = {}
    with tarfile.open(asset, 'w:gz', compresslevel=1, dereference=True) as tar:
        for p in group_files(group):
            name = p.relative_to(ROOT).as_posix()
            files[name] = {'bytes': p.stat().st_size, 'sha256': file_digest(p)}
            tar.add(p, arcname=name, recursive=False)
    manifest = {'schema': 1, 'asset': asset.name, 'bytes': asset.stat().st_size,
                'sha256': file_digest(asset), 'files': files}
    (directory / (group + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{group}: {len(files)} files, {asset.stat().st_size / 1024**2:.1f} MiB', flush=True)

def verify(manifest_path, directory, restore_to=None):
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()) if manifest_path.suffix == '.gz' else manifest_path.read_bytes())
    asset = directory / manifest['asset']
    if asset.stat().st_size != manifest['bytes'] or file_digest(asset) != manifest['sha256']:
        raise ValueError(f'Archive checksum mismatch: {asset}')
    seen = set()
    with tarfile.open(asset, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in manifest['files'] or member.name in seen:
                raise ValueError(f'Unexpected archive member: {member.name}')
            expected = manifest['files'][member.name]
            if member.size != expected['bytes'] or digest(tar.extractfile(member)) != expected['sha256']:
                raise ValueError(f'Member checksum mismatch: {member.name}')
            seen.add(member.name)
    if seen != set(manifest['files']):
        raise ValueError('Missing archive members')
    if restore_to:
        # Preflight every destination before writing; never replace existing data.
        target = restore_to.resolve()
        for name, record in manifest['files'].items():
            dest = (target / name).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f'Unsafe path: {name}')
            if dest.exists() and (not dest.is_file() or file_digest(dest) != record['sha256']):
                raise ValueError(f'Existing file differs: {dest}')
        with tarfile.open(asset, 'r:gz') as tar:
            for member in tar:
                dest = target / member.name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('xb') as out:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        out.write(block)
    print(f'Verified {asset.name}: {len(seen)} members', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pack', 'verify', 'restore'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.command == 'pack':
        if not args.group: parser.error('pack requires --group')
        pack(args.group, args.directory)
    else:
        if not args.manifest: parser.error('verify/restore requires --manifest')
        if args.command == 'restore' and not args.destination: parser.error('restore requires --destination')
        verify(args.manifest, args.directory, args.destination if args.command == 'restore' else None)
