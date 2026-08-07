# Reproducing HAREN-CTC

This document lists every setting needed to reproduce the results, and the exact
commands for the main result and each ablation. All commands assume you are in
the repository root and have run `data_preparation/preprocess.py` (see the
[README](../README.md)).

---

## 1. Data splits

- **Corpus:** DAIC-WOZ (audio + turn-level transcripts).
- **Split:** the **official AVEC-2017** partition — no re-splitting, no k-fold.
  A subject appears in exactly one of train / dev(val) / test.
  - `train` ← `train_split_Depression_AVEC2017.csv`
  - `val`   ← `dev_split_Depression_AVEC2017.csv`
  - `test`  ← full test split CSV (`full_test_split.csv`)
- **Subject-level leakage check:** the training script asserts there is **no
  subject overlap** across train/val/test before training starts and aborts if
  any is found.
- **Utterance sampling (per subject):**
  - train: 18 longest turns if `PHQ8_Binary==0`, 46 longest if `==1`
  - val / test: 20 longest turns
  - turns `< 1.0 s` dropped; segments are cut from the original 16 kHz audio.
- **Reference clip counts** (no-offset variant): train ≈ 2794, val ≈ 700,
  test ≈ 940 utterances. Small differences are possible depending on your copy
  of DAIC-WOZ.
- **Two preprocessing variants:**
  - **no-offset (canonical, headline results):** `preprocess.py` default.
  - **offset (ablation):** add `--offset` to apply four transcript timestamp
    corrections — subjects 318 (+34.0 s), 321 (+3.355 s), 341 (+6.07 s),
    362 (+16.54 s).

---

## 2. Fixed hyper-parameters

| hyper-parameter | value | where set |
|-----------------|-------|-----------|
| learning rate | **1e-5** | `LR` env (default) |
| optimizer | AdamW | `create_opt()` |
| **weight decay** | **1e-4** | `create_opt()` |
| classification loss | `BCEWithLogitsLoss` | `criterion` |
| CTC loss | `nn.CTCLoss(blank=20, zero_infinity=True)` | `ctc_loss_fn` |
| total loss | `cls + ctc_weight · warmup · ctc` | training loop |
| `ctc_weight` | 0.05 | `WavLMClassificationModel(ctc_weight=0.05)` |
| CTC warmup | linear over first **5** epochs (`CTC_WARMUP_EPOCHS`) | training loop |
| CTC classes | `2k+1 = 21`, `k=10`, blank index 20 | model + loss |
| CTC targets | HuBERT-large layer 12, online per-utterance k-means (k=10), run-length collapsed, `+k` token shift for depressed class | `generate_hubert_policy_targets_online` |
| dropout | 0.5 | `DROPOUT` |
| layer groups | 2, exponential init, α=0.95 | `AdaptiveWeightedPool` |
| co-attention heads | 2 | `CoAttentionModule` |
| batch size | 16 | `BATCH_SIZE` env |
| epochs | 15, **no early stopping** | `NUM_EPOCHS` env |
| max clip length | 10.0 s (random crop train / head crop eval) | `AudioDataset(max_sec=10.0)` |
| mixed precision | AMP (`torch.cuda.amp`) | training loop |
| class balancing | `WeightedRandomSampler` seeded per run (disable with `NO_SAMPLER=1`) | training loop |
| backbone | `microsoft/wavlm-large`, frozen weights + eval mode + all stochastic ops zeroed | `_freeze_wavlm_backbone()` |

### Seeds

The five paper seeds (fixed via `random`, `numpy`, `torch`,
`torch.cuda`, `cudnn.deterministic=True`, `cudnn.benchmark=False`):

```
123, 1234, 12345, 123456, 1234567
```

Every metric is reported as **mean ± sd over these 5 seeds** (`ddof=1`),
computed at the best test epoch per seed and written to
`log_fixed_split_summary.csv`.

---

## 3. Main result

No-offset data, LR 1e-5, class-balanced sampler, 5 seeds, 15 epochs:

```bash
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split-nooffset \
    --gpu 0 --run-tag main
```

Raw equivalent:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=./processed_data-utterance-fixed-split-nooffset \
SEEDS=123,1234,12345,123456,1234567 \
NUM_EPOCHS=15 BATCH_SIZE=16 LR=1e-5 RUN_TAG=main \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u src/train_haren_ctc.py
```

Read the summary:

```bash
cat logs_fixed_split_BCE_high_dropout_CTC005_3stra_main/log_fixed_split_summary.csv
```

---

## 4. Ablations

Each ablation just changes a flag / env var; everything else stays fixed.

### 4.1 Learning-rate sweep (with sampler disabled)

```bash
for lr in 5e-6 3e-6 1e-5; do
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --lr $lr --no-sampler --run-tag lr${lr}_nos
done
```

### 4.2 No class-balancing sampler

```bash
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split-nooffset \
    --no-sampler --run-tag main_nosampler
```

### 4.3 Offset vs. no-offset preprocessing

```bash
# build the offset variant once
python data_preparation/preprocess.py \
    --audio-dir /path/to/DAIC/wav_files --trans-dir /path/to/DAIC/transcripts \
    --label-dir /path/to/DAIC/labels \
    --out-root ./processed_data-utterance-fixed-split --offset

# train on it
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split \
    --run-tag offset
```

### 4.4 Synthetic data augmentation (optional)

Merge an extra flat directory of synthetic 16 kHz train wavs (each with
matching `.label` / `.phq_label` sidecars) into the training set only. Because
the synthetic set is already class-balanced, pair it with `--no-sampler`:

```bash
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split-nooffset \
    --cdoa-train-dir /path/to/synthetic_16k \
    --no-sampler --run-tag aug
```

### 4.5 Single-seed smoke test

Fast sanity check (1 seed, 2 epochs) before launching a full run:

```bash
python scripts/run_experiments.py \
    --data-root ./processed_data-utterance-fixed-split-nooffset \
    --seeds 1234 --epochs 2 --run-tag smoke
```

---

## 5. Determinism caveats

Runs are seeded and `cudnn.deterministic=True`, but bit-exact reproducibility
across different GPUs, CUDA/cuDNN versions, or PyTorch builds is not guaranteed
(AMP, non-deterministic CUDA kernels, and driver differences). Expect the
reported mean ± sd to match within noise, not to the last decimal. The reference
environment: Python 3.12, PyTorch 2.7.0+cu126, transformers 4.57.6, single
NVIDIA V100.
