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

## 2. Obtain official data privately

Cloning downloads CODE ONLY. Obtain competition data through the registered team account. Do not publish data, credentials or tokens here. Preserve the official dataset access and redistribution rules.

Place ten rank CSVs under `/content/student_test/training/`:

```text
train_D.csv, train_C.csv, train_B.csv, train_A.csv
train_1D.csv, train_2D.csv, train_3D.csv
train_4D.csv, train_5D.csv, train_6D.csv
```

When available, put six files under `/content/student_test/tests/`:

```text
rank_prediction_test_public.csv
rank_prediction_test_private.csv
player_identification_test_public.csv
player_identification_test_private.csv
player_identification_candidates_public.csv
player_identification_candidates_private.csv
```

Upload/extract authorized data via the Colab Files panel, or mount personal Google Drive and link those folders. Check the archive's internal folder structure after extraction.

## 3. Generate small trial notebooks

```python
%cd /content/student_test
!python prepare_trial.py
```

This generates `trial_rank-prediction.ipynb` and `trial_player-identification.ipynb`. Download a generated file from the Files panel, then upload/open it as a notebook in Colab. Locally, open it in VS Code.

**A separate Colab notebook can have a separate runtime.** Repeat cloning, package installation and private data access there if necessary. Add `%cd /content/student_test` in a new first code cell. Opening a GitHub notebook does not automatically clone its supporting Python files.

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
- `requirements.txt`: helper dependencies.
- `BASELINE_REFERENCE.md`, `README.en.md`: English upstream reference documentation.

Based on [AILAB-NDHU/AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial), whose README credits Serkan Kavak of NDHU AI Lab and permits use, modification and distribution for teaching and competition. Setup and trial materials were added for this group. No independent software license is asserted here.

[AI CUP overview](https://www.aicup.tw/ai-cup-2026-competition) | [Registration](https://go.aicup.tw/) | [Player competition](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688) | [Rank competition](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)

Official rules and the instructor's brief govern participation. This repository registers no accounts and submits no predictions automatically.
