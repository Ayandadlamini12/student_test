# AI CUP 2026: Student Test

English follow-along materials for the Machine Learning group. Both Rank Prediction and Player Identification are compulsory. This is a teaching copy, not the official competition repository.

## 1. Start in Colab

[Open environment setup](https://colab.research.google.com/github/Ayandadlamini12/student_test/blob/main/00_environment_setup.ipynb)

Select **Runtime > Change runtime type > GPU** and run the setup notebook in order. Colab normally supplies PyTorch. The reference baseline uses Python 3.10.12, PyTorch 2.6.0+cu124 and Ubuntu 22.04.5. Other versions require verification rather than assuming compatibility.

```python
!git clone https://github.com/Ayandadlamini12/student_test.git /content/student_test
%cd /content/student_test
%pip install -r requirements.txt
```

If already cloned, skip cloning. Restart the kernel if requested after installation. Check `torch.cuda.is_available()` and the GPU name before training. The requirements file installs helper packages, not PyTorch.

## 2. Download the official training data

The organizer publishes training data in a separate public repository: [AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/tag/v1.0). The tutorial-code repository itself does not contain the data.

Run this in the cloned repository:

```python
%cd /content/student_test
!python download_training_data.py
```

The helper downloads the pinned v1.0 archive (224,515,476 bytes, about 214 MiB), checks the organizer's SHA-256, safely extracts it, and validates all ten CSVs: 100,000 games each, required columns, rank labels, colors and nonempty SGFs. Existing training folders are validated without overwriting them. Allow space for the archive, approximately 1.3 GB of extracted CSVs, and later feature caches/checkpoints.

Expected files: `training/train_D.csv`, `train_C.csv`, `train_B.csv`, `train_A.csv`, and `train_1D.csv` through `train_6D.csv`.

Organizer checksum:

```text
959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b
```

Data stays excluded from this teaching repository. The organizer restricts use to the 2026 AI CUP competition and prohibits identifying real users. No AI CUP login is needed to download this public training release. Team registration remains necessary for competition participation.

**Test and candidate files are separate.** Obtain the official release through the competition platform when available and place these under `tests/`:

```text
rank_prediction_test_public.csv
rank_prediction_test_private.csv
player_identification_test_public.csv
player_identification_test_private.csv
player_identification_candidates_public.csv
player_identification_candidates_private.csv
```

The helper downloads training data only. It does not create test answers, register a team or submit predictions.

## 3. Generate small trial notebooks

```python
%cd /content/student_test
!python prepare_trial.py
```

Run the cell above in the setup notebook after Steps 1 and 2. It creates two files inside `/content/student_test/`:

- `trial_rank-prediction.ipynb`: run this one first.
- `trial_player-identification.ipynb`: run this after the Rank trial works.

### 3A. Download and open the generated notebook

1. Click the folder icon on the left of Colab to open **Files**.
2. Expand `student_test`. Refresh the file list if the generated notebooks are not visible.
3. Open the three-dot menu beside `trial_rank-prediction.ipynb` and choose **Download**.
4. Select **File > Open notebook > Upload** and upload that downloaded `.ipynb` file. Uploading the notebook does not upload `utils.py`, `network.py` or the training CSVs.
5. In the opened trial notebook, select **Runtime > Change runtime type > GPU**.

### 3B. Prepare the trial notebook's own runtime BEFORE imports

A different notebook can have a different runtime and filesystem. Do not assume that files or packages from the setup notebook are present. Click **+ Code**, move the new cell ABOVE the original imports cell, and paste:

```python
from pathlib import Path
import subprocess
import os
import sys

repo = Path('/content/student_test')
if not (repo / '.git').exists():
    subprocess.run(['git', 'clone',
                    'https://github.com/Ayandadlamini12/student_test.git',
                    str(repo)], check=True)
else:
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)

os.chdir(repo)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

for name in ['utils.py', 'network.py', 'download_training_data.py']:
    print(name, 'FOUND' if (repo / name).is_file() else 'MISSING')
```

Run that cell first. All three files should print **FOUND**. This also updates an older clone that does not yet contain the download script. If Git reports local edits and refuses to update, preserve those edits before resolving the update; do not delete the folder blindly.

### 3C. Install packages and make data available in this runtime

Add and run these cells below the new setup cell, still BEFORE the original imports:

```python
%pip install -r /content/student_test/requirements.txt
```

```python
subprocess.run([sys.executable, '-u',
                '/content/student_test/download_training_data.py'], check=True)
```

The downloader validates an existing training folder instead of downloading it again. In a new runtime, it downloads and extracts the official archive. Wait for the ten validation messages and the final readiness message. If installation requests a restart, restart and rerun the setup cell before continuing.

### 3D. Run the actual trial in order

1. Rerun the original imports cell. `from utils ...` and `from network ...` should now work.
2. Run the configuration cell and check that the device is CUDA and `NUM_EPOCHS` is 1.
3. Run data loading, feature extraction, Data Loader and Model Training cells in order.
4. Stop before **Section 4: Inference** unless the official test/candidate files are available.
5. Save the notebook, trial screenshots, history and checkpoints. Repeat 3A–3D for the Player trial, checking its own runtime too.

If `ModuleNotFoundError: No module named 'utils'` appears, rerun 3B and check that the working directory is `/content/student_test`. **Do not install a package named `utils`: the required module is this repository's `utils.py`.**

If the download command returns `CalledProcessError`, inspect the script's error text immediately before the traceback. A missing script often indicates an old clone or the wrong working directory. The absolute script path above avoids the latter.

### Local VS Code alternative

Run `python prepare_trial.py` in the repository's terminal, then open `trial_rank-prediction.ipynb` directly. Select the correct local Python kernel and run from the repository folder. Skip Colab-only `/content` paths and `%pip` installation if dependencies already exist in that environment.

| Trial setting | Rank | Player |
|---|---|---|
| Data | First 1,000 rows per rank file | First 1,000 rows per rank file, followed by player filtering |
| Epochs | 1 | 1 |
| Batch size | 16 | 8 |
| Validation | 1,000 held-out games | No held-out split added to upstream player baseline |
| Steps | One pass through trial training records | 20 smoke-test steps |
| Extraction | Serial | Serial |
| DataLoader workers | 0 | 0 |

The first rows are a convenient demo subset, not a representative validation design. Player sampling requires at least TWO eligible players with at least TWO games each. Check the printed counts. If too few survive, select a player-aware subset or increase the sample size; do not lower the two-game requirement.

Trial features/checkpoints use separate `trial-` paths. The original baseline notebooks are unchanged apart from English translation. Repeated trials reuse their trial paths: clear or rename trial caches when data or feature configuration changes.

Run cells in order through **Section 3: Model Training**. Confirm nonempty features, expected shapes, finite loss and saved checkpoints/history. Rank outputs ten logits; Player outputs 256-dimensional embeddings. Save environment and trial screenshots.

## 4. Inference and submission

Section 4 needs official test/candidate files. Stop after training if unavailable. Training alone does not establish end-to-end submission success.

Use the best checkpoint and produce:

- Rank: `question_id,pred_rank`.
- Player: `question_id,top_1,top_2,top_3,top_4,top_5`.

Each CSV needs 800 prediction rows: 400 public plus 400 private. Preserve actual IDs and prefixes. Player IDs must be unique per row and belong to the matching candidate pool. Tiny-trial scores will not reproduce the full reference public scores of 0.4631 (Rank) and 0.2845 (Player).

## 5. Full baseline after successful trials

[Open Rank baseline](https://colab.research.google.com/github/Ayandadlamini12/student_test/blob/main/rank-prediction.ipynb)

[Open Player baseline](https://colab.research.google.com/github/Ayandadlamini12/student_test/blob/main/player-identification.ipynb)

These retain full upstream settings. Run one training notebook at a time. Adjust batch sizes to the available GPU. Reference memory figures: Rank about 9 GB VRAM; Player about 4 GB. Feature storage: about 1.7 GB and 15 GB respectively, plus raw data, checkpoints and repeated runs.

Full notebooks use Linux multiprocessing with notebook-defined workers. Windows requires adaptation. Generated trials bypass extraction processes and use zero DataLoader workers. Full notebooks are not automatically Windows-safe.

The player baseline selects checkpoints by TRAINING loss and does not provide held-out validation. Document that limitation and meet course evidence requirements without calling training loss validation loss.

## 6. Save results and assignment evidence

Colab `/content` files are temporary. Download checkpoints, history, notebooks and CSVs or copy them to private Google Drive. Saved notebook output does not preserve model files.

Retain environment details, trial screenshots, full-training curves, timing, best-epoch metrics, submitted CSVs and leaderboard evidence. Make one controlled change per task, rerun and submit, then explain the comparison. Improvement is optional.

Submit one 3–5-page report, code or repository link, screenshots, CSVs, reflection and contributions. One member uploads `HW_AICUP_<team number>_<student ID>.zip`. The supplied course brief lists 10 November 2026 at 12:00 AM; aim to upload on 9 November. Competition dates are separate.

## Troubleshooting

| Problem | Check |
|---|---|
| Cannot import utils/network | Set working directory to repository root |
| Missing sgfmill/webdataset | Install requirements in active runtime |
| CUDA unavailable | Select GPU runtime and verify allocation |
| Out of memory | Reduce batch size and stop other training sessions |
| No objects to concatenate | Verify ten training CSVs and configured path |
| Player sampling error | Check two players survive the two-game filter |
| Missing tests | Stop before inference until official files are available |
| Windows process failure | Use serial trials; adapt full notebooks separately |

## Files and attribution

- `00_environment_setup.ipynb`: explained environment checks and setup.
- Both baseline notebooks: translated explanations and displayed messages.
- `network.py`, `utils.py`: original model and feature code.
- `prepare_trial.py`: generates separate small-trial notebooks.
- `download_training_data.py`: downloads, verifies and validates official training data.
- `requirements.txt`: helper dependencies.
- `BASELINE_REFERENCE.md`, `README.en.md`: English upstream reference documentation.

Based on [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial), whose README credits Serkan Kavak of NDHU AI Lab and permits use, modification and distribution for teaching and competition. Setup and trial materials were added for this group. No independent software license is asserted here.

[AI CUP overview](https://www.aicup.tw/ai-cup-2026-competition) | [Registration](https://go.aicup.tw/) | [Player competition](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688) | [Rank competition](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)

Official rules and the instructor's brief govern participation. This repository registers no accounts and submits no predictions automatically.
