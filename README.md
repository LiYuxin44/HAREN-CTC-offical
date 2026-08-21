# HAREN-CTC

HAREN-CTC is a speech-based depression-detection model built on frozen
WavLM-Large representations. It learns two soft groups over the 24 WavLM
layers, fuses them with co-attention, and predicts depression at subject level
by averaging utterance probabilities. The reported fixed-split workflow uses
corrected timestamp offsets and its legacy local-unit CTC head; the sole
reported 5-fold result uses the repaired neutral-unit CTC head.

## Reported results

All values are subject-level at threshold 0.5 without a seed ensemble. Mean
rows report mean ± sample SD; the fixed Dev maximum is one selected checkpoint.

| evaluation | seeds | epoch policy | Macro-F1 | ROC-AUC |
|---|---:|---|---:|---:|
| official fixed Dev mean, corrected offset + CTC | 5 | per-seed best Dev checkpoint | **0.6103 ± 0.0470** | 0.5841 ± 0.0523 |
| official fixed Dev max, corrected offset + CTC | 1 | best seed/epoch selected by Dev Macro-F1 | **0.6686** | 0.5978 |
| official fixed Test, corrected offset + CTC | 5 | corresponding frozen Dev checkpoints | **0.5446 ± 0.0442** | 0.5134 ± 0.0351 |
| PHQ-balanced 5-fold OOF, offset + CTC | 5 | shared test-selected epoch 11; test-selected seeds | **0.5676 ± 0.0092** | 0.5476 ± 0.0134 |

For the fixed split, 17 CTC settings were screened with seed 123 on Dev only.
Three finalists were then confirmed on the predeclared seeds `123, 1234,
12345, 123456, 1234567`. The frozen winner uses local-unit CTC with `K=10`, a
five-epoch warmup, and weight `0.005`. Its five checkpoints score Dev Macro-F1
`0.6103 ± 0.0470` and Dev ROC-AUC `0.5841 ± 0.0523`; the table gives their
corresponding Test scores. The highest-Dev checkpoint is seed 123 at epoch 7:
Dev Macro-F1 `0.6686` and Dev ROC-AUC `0.5978`; its corresponding Test
Macro-F1 is `0.5761`.

The CTC setting and checkpoints did not use Test values for selection.
However, this repository's official fixed Test split had already been examined
in earlier experiments, so `0.5446` is not an independent untouched-test
estimate.

The 5-fold result is the single CV reporting convention used by this
repository: all 189 subjects are split once into five outer folds balanced over
PHQ-8 bins `0–4`, `5–9`, `10–14`, `15–19`, and `20–24`; each model trains on
the other four folds; results are pooled over all five held-out folds within
each seed and then summarized across five seeds.

Important: epoch 11 was selected after inspecting the same fold-test
trajectories. The reported five seeds were then selected post hoc from ten
candidate seeds by their epoch-11 fold-test Macro-F1. The 5-fold number is
therefore a doubly test-tuned exploratory estimate, not independent test
performance. Its mean is optimistically biased and its SD is artificially
reduced; it must not be compared directly with the official fixed-split result
as if both used the same selection protocol.

## Model

```text
raw waveform
    │
    ▼
frozen WavLM-Large (24 layers)
    │
    ▼
adaptive two-group layer pooling
    │
    ▼
co-attention + [CLS]
    ├──────────────► depression classifier (BCE)
    └──────────────► optional temporal head (CTC / frame CE)
```

The fixed-split headline retains the legacy preprocessing, local-unit CTC,
crop, AdamW, FP16, and head-view settings, with the frozen CTC weight `0.005`.
The PHQ-balanced workflow uses valid-audio-first WavLM normalization, true-length
attention masks, deterministic epoch-keyed crops, AdamW with no-decay parameter
groups and warmup/cosine scheduling, and deterministic head/center/tail
evaluation views. Its CTC branch uses fold-local, outer-train-only neutral
HuBERT units (`K=10`, stride 1) and a shared-gradient target ratio of `0.0001`.

## Repository layout

```text
data_preparation/
  preprocess.py                          fixed-split preprocessing
  prepare_phq_stratified_train_test.py   audited PHQ-balanced manifests
  prepare_phq_ctc_units.py               fold-local K=10 HuBERT units
artifacts/fixed_default/result.json       fixed-split metrics and checkpoint hashes
artifacts/phq5_default/                   canonical result + one checkpoint
scripts/
  run_experiments.py                     single-run CLI
  eval_checkpoints.py                    frozen fixed-split test evaluation
  run_fixed_offset_ctc.py                selected five-seed fixed runner
  run_phq_balanced_cv.py                 selected 5-seed × 5-fold runner
  summarize_phq_balanced_cv.py           fixed epoch-11 OOF summary
src/
  train_haren_ctc.py                     model and training loop
  haren_ctc_utils.py                     mask, target, and loss helpers
  training_stability.py                  deterministic data/training helpers
  global_unit_cache.py                   optional neutral-unit cache reader
tests/                                    unit and protocol tests
docs/REPRODUCE.md                         complete reproduction commands
```

## Installation

Python 3.12 is used by the reference environment.

```bash
conda create -n haren-ctc python=3.12 -y
conda activate haren-ctc

# Adjust the CUDA wheel index to your machine.
pip install torch==2.7.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`ffmpeg` is required by `pydub` during preprocessing. WavLM and HuBERT weights
are downloaded from Hugging Face on first use.

## Data

Request DAIC-WOZ from <https://dcapswoz.ict.usc.edu/>. For every participant,
provide the session waveform, transcript, and AVEC-2017 labels. The corrected
metadata must satisfy `PHQ8_Binary == (PHQ8_Score >= 10)`; in particular,
participant 409 has binary label 1.

Fixed-split preprocessing:

```bash
python data_preparation/preprocess.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --out-root ./datasets/fixed_corrected_offset
```

PHQ-balanced all-189 preprocessing:

```bash
python data_preparation/prepare_phq_stratified_train_test.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --out-root ./datasets/phq5
```

The PHQ workflow consumes the official train, dev, and test subjects as one
189-subject CV pool. It does not preserve an independent official test set.

## Run

A single fixed-split development run from the pinned five-seed protocol:

```bash
python scripts/run_fixed_offset_ctc.py \
  --data-root ./datasets/fixed_corrected_offset \
  --output-root ./runs/fixed \
  --gpu 0 --seeds 123
```

The five tuned Dev-selected checkpoints are retained locally under
`checkpoints/fixed_default/`. This directory is deliberately
gitignored; their names, sizes, and SHA-256 hashes are recorded in
`artifacts/fixed_default/result.json`.

The complete 25-run PHQ-balanced protocol:

```bash
python data_preparation/prepare_phq_ctc_units.py \
  --manifest-root ./datasets/phq5 \
  --output-root ./datasets/phq5_units

python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --unit-root ./datasets/phq5_units \
  --output-root ./runs/phq5

python scripts/summarize_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --runs-root ./runs/phq5 \
  --output-root ./results/phq5
```

The canonical numbers and one representative trainable-head checkpoint are in
`artifacts/phq5_default/`. The checkpoint is seed 2026 / fold 4 / epoch 11,
the historically highest individual fold among the 25 reported seed-fold
models. It is not a substitute for the five fold checkpoints required to
reproduce one complete OOF seed.

Use `--folds` and `--seeds` to partition the 25 jobs across GPUs. Each
fold/seed output directory is immutable: the runner refuses to overwrite
nonempty results.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for exact seeds, invariants,
audits, and frozen-checkpoint evaluation.

## License

Released under the [MIT License](LICENSE). DAIC-WOZ data and the upstream
WavLM/HuBERT weights retain their respective licenses and are not distributed
here.
