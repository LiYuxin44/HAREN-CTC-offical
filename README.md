# HAREN-CTC

HAREN-CTC is a speech-based depression-detection model built on frozen
WavLM-Large representations. It learns two soft groups over the 24 WavLM
layers, fuses them with co-attention, and predicts depression at subject level
by averaging utterance probabilities. The temporal CTC head remains available
as an optional research component; the reported headline models below are
BCE-only.

## Reported results

All values are subject-level mean ± sample SD at threshold 0.5, without a
seed ensemble.

| evaluation | seeds | epoch policy | Macro-F1 | ROC-AUC |
|---|---:|---|---:|---:|
| official fixed split, corrected/no-offset | 5 | dev-selected checkpoint | **0.5764 ± 0.0361** | 0.5281 ± 0.0536 |
| PHQ-balanced 5-fold OOF | 5 | shared test-selected epoch 14; test-selected seeds | **0.5742 ± 0.0036** | 0.5536 ± 0.0150 |

The 5-fold result is the single CV reporting convention used by this
repository: all 189 subjects are split once into five outer folds balanced over
PHQ-8 bins `0–4`, `5–9`, `10–14`, `15–19`, and `20–24`; each model trains on
the other four folds; results are pooled over all five held-out folds within
each seed and then summarized across five seeds.

Important: epoch 14 was selected after inspecting the same fold-test
trajectories. The reported five seeds were then selected post hoc from ten
candidate seeds by their epoch-14 fold-test Macro-F1. The 5-fold number is
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

The selected training path uses valid-audio-first WavLM normalization,
true-length attention masks, deterministic epoch-keyed crops, AdamW with
no-decay parameter groups and warmup/cosine scheduling, and deterministic
head/center/tail evaluation views.

## Repository layout

```text
data_preparation/
  preprocess.py                          fixed-split preprocessing
  prepare_phq_stratified_train_test.py   audited PHQ-balanced manifests
scripts/
  run_experiments.py                     single-run CLI
  eval_checkpoints.py                    frozen fixed-split test evaluation
  run_phq_balanced_cv.py                 selected 5-seed × 5-fold runner
  summarize_phq_balanced_cv.py           fixed epoch-14 OOF summary
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
  --out-root ./datasets/fixed_corrected_nooffset
```

PHQ-balanced all-189 preprocessing:

```bash
python data_preparation/prepare_phq_stratified_train_test.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --offset \
  --out-root ./datasets/phq5
```

The PHQ workflow consumes the official train, dev, and test subjects as one
189-subject CV pool. It does not preserve an independent official test set.

## Run

A single fixed-split development run:

```bash
python scripts/run_experiments.py \
  --data-root ./datasets/fixed_corrected_nooffset \
  --gpu 0 --seeds 1234 --epochs 15 \
  --lr 1e-4 --batch-size 16 --weight-decay 1e-5 --dropout 0.3 \
  --ctc-enabled 0 --test-policy none \
  --wavlm-mask-policy true_length \
  --wavlm-preprocess-policy valid_then_pad \
  --train-crop-policy epoch_keyed \
  --optimizer-policy no_decay_warmup_cosine \
  --eval-crop-policy multi3 \
  --output-dir ./runs/fixed/seed1234
```

The complete 25-run PHQ-balanced protocol:

```bash
python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --output-root ./runs/phq5

python scripts/summarize_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --runs-root ./runs/phq5 \
  --output-root ./results/phq5
```

Use `--folds` and `--seeds` to partition the 25 jobs across GPUs. Each
fold/seed output directory is immutable: the runner refuses to overwrite
nonempty results.

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for exact seeds, invariants,
audits, and frozen-checkpoint evaluation.

## License

Released under the [MIT License](LICENSE). DAIC-WOZ data and pretrained model
weights retain their respective licenses and are not distributed here.
