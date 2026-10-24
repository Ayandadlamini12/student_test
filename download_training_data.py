"""Download, verify and extract the organizer's v1.0 training release."""
import csv
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/download/v1.0/aicup2026_go_training.tar.xz'
SHA256 = '959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b'
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']


def validate(folder):
    """Check the schema, rank labels and row count of each released CSV."""
    for level in LEVELS:
        path = folder / f'train_{level}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'player_id', 'game_id', 'rank', 'color', 'sgf_content'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'Unexpected schema: {path.name}')
            count = 0
            for row in reader:
                if row['rank'] != level or row['color'] not in ('B', 'W'):
                    raise ValueError(f'Invalid rank/color in {path.name}, row {count + 1}')
                if not row['sgf_content'].strip():
                    raise ValueError(f'Empty SGF in {path.name}')
                count += 1
            if count != 100_000:
                raise ValueError(f'{path.name}: expected 100,000 rows; got {count:,}')
        print(f'Validated {path.name}: {count:,} games', flush=True)


def main():
    target = ROOT / 'training'
    if target.exists():
        print('Existing training folder: validating without overwriting.', flush=True)
        validate(target)
        return
    cache = ROOT / 'downloads'
    cache.mkdir(exist_ok=True)
    archive = cache / 'aicup2026_go_training.tar.xz'
    if not archive.exists():
        partial = archive.with_suffix('.xz.part')
        print('Downloading official training archive (about 214 MiB)...', flush=True)
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open('wb') as sink:
            shutil.copyfileobj(source, sink)
        partial.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError('SHA-256 mismatch. Remove the cached archive and retry.')
    print('SHA-256 matches the organizer checksum.', flush=True)
    with tempfile.TemporaryDirectory(prefix='aicup-extract-', dir=ROOT) as temp:
        destination = Path(temp)
        with tarfile.open(archive, 'r:xz') as bundle:
            for member in bundle.getmembers():
                resolved = (destination / member.name).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe archive path')
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Archive contains a link or special file')
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(destination, filter="data")
            else:
                bundle.extractall(destination)
        extracted = destination / 'training'
        validate(extracted)
        extracted.rename(target)
    print('Ready: 1,000,000 training games. Official test files are separate.', flush=True)


if __name__ == '__main__':
    main()
