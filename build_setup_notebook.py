"""Generate the explained Colab setup notebook without running its cells."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []

def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True), id=f'setup-{len(cells)}'))

def code(text):
    cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(True), id=f'setup-{len(cells)}'))

md('''# AI CUP: Group Environment Setup
Select **Runtime > Change runtime type > GPU** in Colab. Run cells in order.
This notebook installs helper dependencies and creates trial notebooks. It does not start training or download competition data.
''')
md('''## 1. Obtain teaching files
Responsibility: clone the public repository if absent and set the working directory.
Parameter: repository path. Output: local working directory.
Local VS Code users can run this notebook from the cloned folder.
''')
code('''from pathlib import Path
import subprocess
import sys
import os
repo = Path('/content/student_test') if Path('/content').exists() else Path.cwd()
if not (repo / 'requirements.txt').exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/Ayandadlamini12/student_test.git',
                    str(repo)], check=True)
os.chdir(repo)
print('Working directory:', Path.cwd())
''')
md('''## 2. Install helper dependencies
Responsibility: install requirements into the active kernel. Parameter: requirements.txt.
Output: installer status. Colab normally supplies PyTorch; this cell does not replace it.
Restart the runtime if installation requests it.
''')
code('''subprocess.run([sys.executable, '-m', 'pip', 'install',
                '-r', 'requirements.txt'], check=True)
''')
md('''## 3. Verify packages and GPU
Responsibility: inspect Python, package imports and CUDA access.
Parameters: module names and GPU 0. Output: import status, versions and device information.
A CPU-only pass does not establish GPU readiness.
''')
code('''import importlib
import platform
print('Python:', sys.version)
print('System:', platform.platform())
for name in ['torch', 'numpy', 'pandas', 'matplotlib', 'webdataset', 'sgfmill']:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, '__version__', 'import OK'))
    except Exception as error:
        print('REVIEW:', name, error)
import torch
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
''')
md('''## 4. Tiny computation check
Responsibility: test forward/backward execution. Parameters: a 4-by-4 tensor and device.
Output: loss and finite-gradient status. This is not model training.
''')
code('''device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.ones((4, 4), device=device, requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
print('Device:', device, '| loss:', loss.item())
print('Finite gradients:', bool(torch.isfinite(x.grad).all().item()))
del x, loss
''')
md('''## 5. Obtain authorized data privately
Place the ten rank CSVs in training/ and six test/candidate CSVs in tests/ when available.
Upload/extract the official files or mount private Google Drive. Do not publish data or credentials.
Responsibility: check expected training files and disk space. Parameters: rank labels.
Output: file presence and free GiB. Features alone require about 1.7 GB for Rank and 15 GB for Player.
''')
code('''import shutil
levels = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
for level in levels:
    path = repo / 'training' / f'train_{level}.csv'
    print(path.name, 'FOUND' if path.is_file() else 'MISSING')
print('Free disk GiB:', round(shutil.disk_usage(repo).free/1024**3, 1))
''')
md('''## 6. Generate small trials
Responsibility: create separate trial notebooks. Parameter: prepare_trial.py.
Output: two trial files; no training starts. Trials use 1,000 rows per rank and one epoch.
Player training is capped at 20 steps and requires two eligible players with two games each.
''')
code('''subprocess.run([sys.executable, 'prepare_trial.py'], check=True)
for path in sorted(repo.glob('trial_*.ipynb')):
    print(path.name)
''')
md('''## Continue with the team
Download a generated trial from the Files panel and upload/open it in Colab, or open it directly in VS Code.
A different Colab notebook may have a separate runtime: repeat clone/install/data access as needed.
Add `%cd /content/student_test` before baseline cells. Run through Section 3 first.
Section 4 needs official test files; stop before inference when they are unavailable.
Save outputs and checkpoints privately before the temporary runtime disappears.
The original baseline files retain full-training settings. See README.md for full guidance.

Source: [Official NDHU baseline](https://github.com/AILAB-NDHU/AICup-2026-Tutorial).
''')
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'}, 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
(root / '00_environment_setup.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
