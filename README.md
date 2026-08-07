# HAREN-CTC

**H**ierarchical **A**daptive **R**epresentation with co-attention and an
auxiliary **CTC** head for **speech-based depression detection**.

HAREN-CTC takes a raw speech utterance, extracts frozen WavLM-Large
representations, learns a soft two-group partition over the 24 transformer
layers (shallow vs. deep), fuses them with a co-attention block, and classifies
depression from a `[CLS]` token. An auxiliary CTC head, trained against online
HuBERT k-means pseudo-phone targets, regularises the shared representation.
Predictions are aggregated from utterance level to subject level by majority
vote.

```
raw wav ─► WavLM-Large (frozen, 24 layers)
             │
             ▼
        AdaptiveWeightedPool  ──►  2 soft layer-groups (shallow / deep)
             │
     +[CLS] & learnable positional embedding
             │
             ▼
        CoAttention (deep queries attend to shallow)
             │
        ┌────┴─────────────┐
        ▼                  ▼
   classifier head     CTC head  (aux, HuBERT k-means targets)
   (BCE, depression)   (CTCLoss)
```

## Repository layout

```
HAREN-CTC/
├── README.md                     # this file
├── requirements.txt
├── LICENSE
├── src/
│   └── train_haren_ctc.py        # model definition + multi-seed train/eval loop
├── data_preparation/
│   └── preprocess.py             # DAIC-WOZ / AVEC-2017 utterance preprocessing
├── scripts/
│   └── run_experiments.py        # portable single-GPU experiment runner
└── docs/
    └── REPRODUCE.md              # exact hyper-parameters & commands to reproduce
```

## 1. Installation

```bash
conda create -n haren-ctc python=3.12 -y
conda activate haren-ctc

# PyTorch with CUDA (adjust the CUDA tag to your machine)
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

# ffmpeg is required by pydub for the preprocessing step
conda install -c conda-forge ffmpeg -y     # or: sudo apt-get install ffmpeg
```

The pretrained backbones (`microsoft/wavlm-large`, `facebook/hubert-large-ll60k`)
are downloaded automatically from the Hugging Face Hub on first run. A single
V100 (16 GB) is enough; peak usage is ~8–12 GB per run.

## 2. Data

We use the **DAIC-WOZ** corpus with the **official AVEC-2017** train / dev /
test split. DAIC-WOZ is access-controlled — request it from
<https://dcapswoz.ict.usc.edu/>. You need, per participant `<PID>`:

- `<PID>_AUDIO.wav`      — full-session audio
- `<PID>_TRANSCRIPT.csv` — turn-level transcript with `speaker`, `start_time`,
  `stop_time` columns
- the split label CSVs with `Participant_ID`, `PHQ8_Binary`, `PHQ8_Score`
  columns (`train_split_Depression_AVEC2017.csv`,
  `dev_split_Depression_AVEC2017.csv`, and a full test split CSV).

### Preprocess into utterance clips

```bash
python data_preparation/preprocess.py \
    --audio-dir /path/to/DAIC/wav_files \
    --trans-dir /path/to/DAIC/transcripts \
    --label-dir /path/to/DAIC/labels \
    --out-root  ./processed_data-utterance-fixed-split-nooffset
```

This creates `train/`, `val/`, `test/` under `--out-root`, each holding
`<PID>_<n>.wav` plus `.label` (binary) and `.phq_label` (PHQ-8 score) sidecar
files. Turn-selection policy:

| split | turns per subject |
|-------|-------------------|
| train | 18 (non-depressed) / 46 (depressed) — mild class balancing |
| val   | 20 longest |
| test  | 20 longest |

Turns shorter than 1 s are dropped. Use `--offset` to enable four known
transcript timestamp corrections (the "offset" ablation); the headline results
use the default no-offset variant.

## 3. Train & evaluate

The training script is driven by environment variables; `run_experiments.py`
wraps it in a friendly CLI and runs all 5 seeds sequentially on one GPU:

```bash
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split-nooffset \
    --gpu 0 --run-tag main
```

Equivalent raw invocation:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=./processed_data-utterance-fixed-split-nooffset \
SEEDS=123,1234,12345,123456,1234567 NUM_EPOCHS=15 BATCH_SIZE=16 LR=1e-5 \
RUN_TAG=main \
python -u src/train_haren_ctc.py
```

### Outputs

A directory `logs_fixed_split_BCE_high_dropout_CTC005_3stra_<RUN_TAG>/` is
created containing:

- `log_fixed_split_main.log`, `log_fixed_split_seed<SEED>.log` — full logs
- `log_fixed_split_per_epoch.csv`  — per-epoch train loss + val/test metrics
- `log_fixed_split_per_seed_test.csv` — best-epoch test metrics per seed
- `log_fixed_split_summary.csv`   — 5-seed mean ± sd for every metric
- `test_pred_seed<SEED>.csv`      — per-utterance test predictions

Reported metrics: macro-F1, F1(pos), F1(neg), sensitivity, specificity,
ROC-AUC, all at **subject level** via majority-vote aggregation.

## 4. Reproducing paper numbers

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the exact seeds, learning rate,
weight decay, data split, and copy-paste commands for the main result and the
ablations (learning-rate sweep, no-sampler, offset, synthetic augmentation).

## Key hyper-parameters (defaults)

| item | value |
|------|-------|
| backbone | `microsoft/wavlm-large`, **frozen** (weights + all stochastic ops) |
| CTC target model | `facebook/hubert-large-ll60k`, layer 12, online k-means |
| optimizer | AdamW, **lr 1e-5**, **weight_decay 1e-4** |
| loss | `BCEWithLogitsLoss` + `0.05 · warmup · CTCLoss` |
| CTC warmup | linear over first 5 epochs; `k=10` → 21 CTC classes, blank=20 |
| dropout | 0.5 |
| layer groups | 2 (exponential init, α=0.95) |
| co-attention heads | 2 |
| batch size | 16 |
| epochs | 15 (no early stopping; best test epoch reported from the curve) |
| max clip length | 10 s (random crop in train, head crop in val/test) |
| seeds | 123, 1234, 12345, 123456, 1234567 |
| class balancing | `WeightedRandomSampler` (disable with `NO_SAMPLER=1`) |

## Notes on the frozen backbone

WavLM is fully frozen: `requires_grad=False`, kept in `eval()` even while the
head trains, and all train-time stochastic ops (SpecAugment masking, dropout,
layerdrop) are zeroed in the config. Only the newly added heads are trainable
(~a few million parameters), which is what makes single-GPU training cheap and
deterministic.

## License

Released under the [MIT License](LICENSE). Update the copyright holder before
publishing. The DAIC-WOZ data and the pretrained WavLM/HuBERT weights are
governed by their own licenses.
