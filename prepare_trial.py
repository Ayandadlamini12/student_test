"""Generate separate one-epoch demonstration notebooks from the English baselines."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for task in ('rank', 'player'):
    name = 'rank-prediction.ipynb' if task == 'rank' else 'player-identification.ipynb'
    nb = json.loads((ROOT / name).read_text(encoding='utf-8'))
    changes = {
        'pd.read_csv(p)': 'pd.read_csv(p, nrows=1000)',
        'NUM_EPOCHS = 5': 'NUM_EPOCHS = 1',
        'NPROC = 8': 'NPROC = 1',
        'BATCH_SIZE = 256': 'BATCH_SIZE = 16',
        'BATCH_SIZE = 100': 'BATCH_SIZE = 8',
        'VAL_SIZE = 100_000': 'VAL_SIZE = 1000',
        'STEPS_PER_EPOCH = 15_000': 'STEPS_PER_EPOCH = 20',
        'LOG_INTERVAL = 5_000': 'LOG_INTERVAL = 5',
        'GPU_BATCH = 4096': 'GPU_BATCH = 128',
        'GPU_BATCH = 64': 'GPU_BATCH = 16',
        'pick_workers(val_shards)': 'pick_workers(val_shards, candidate=0)',
        './player_index.json': './trial-player-index.json',
    }
    for folder in ('rank-features', 'player-features', 'rank-trained-models', 'player-trained-models'):
        changes['./' + folder] = './trial-' + folder
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for before, after in changes.items():
            source = source.replace(before, after)
        if 'p = Process(target=extract_worker' in source:
            lines = source.splitlines()
            start = next(i for i, line in enumerate(lines) if 'p = Process(target=extract_worker' in line)
            old = lines[start:start + 3]
            assert 'p.start()' in old[1] and 'procs.append(p)' in old[2]
            indent = old[0][:len(old[0]) - len(old[0].lstrip())]
            args = 'df.iloc[i::NPROC], i, shard' if task == 'player' else 'df_part.iloc[i::NPROC], pat, i'
            lines[start:start + 3] = [indent + 'if NPROC == 1:', indent + '    extract_worker(' + args + ')', indent + 'else:'] + ['    ' + line for line in old]
            source = '\n'.join(lines) + '\n'
        if task == 'player':
            source = source.replace('num_workers=NPROC,', 'num_workers=0,')
            source = source.replace('persistent_workers=True', 'persistent_workers=False')
        cell['source'] = source.splitlines(True)
        cell['execution_count'] = None
        cell['outputs'] = []
    notice = ('# Small trial only\nReads the first 1,000 rows per rank file, trains for one epoch, '
              'and uses separate trial output paths. Player training is capped at 20 steps. '
              'Run through training first. Inference requires official test files. '
              'Trial scores are not expected to match the full baseline.\n')
    nb['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=notice.splitlines(True), id='trial-notice'))
    out = ROOT / ('trial_' + name)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Created', out.name)
print('Player trial needs two eligible players, each with at least two games.')
print('If too few survive the filter, use a larger or player-aware subset.')
