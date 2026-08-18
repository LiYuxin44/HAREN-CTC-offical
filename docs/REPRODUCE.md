# Reproducing the reported results

Commands assume the repository root as the working directory. Generated audio,
manifests, checkpoints, predictions, and result tables are intentionally not
versioned.

## 1. Environment

```bash
conda create -n haren-ctc python=3.12 -y
conda activate haren-ctc
pip install torch==2.7.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Install `ffmpeg` separately for audio preprocessing. The reference WavLM
revision used by the PHQ-balanced runs is
`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.

## 2. Label and data invariants

Expected AVEC-2017 metadata:

| source split | subjects |
|---|---:|
| train | 107 |
| dev | 35 |
| test | 47 |
| total | 189 |

The label files must provide `Participant_ID`, `PHQ8_Binary`, and
`PHQ8_Score`. The preprocessing code enforces:

- unique participants and disjoint official splits;
- `PHQ8_Binary == (PHQ8_Score >= 10)`;
- corrected binary label 1 for participant 409;
- contiguous clip IDs and matching `.label` / `.phq_label` sidecars;
- 18 train clips for a negative subject, 46 for a positive subject, and
  20 evaluation clips per subject.

## 3. Official fixed split

Build the corrected no-offset data:

```bash
python data_preparation/preprocess.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --out-root ./datasets/fixed_corrected_nooffset
```

The reported BCE-only winner uses:

- seeds `123, 1234, 12345, 123456, 1234567`;
- learning rate `1e-5`;
- batch size `16`;
- weight decay `1e-5`;
- dropout `0.1`;
- 15 epochs;
- the legacy preprocessing/crop/AdamW defaults;
- checkpoint selection on dev Macro-F1, then dev AUC, F1(pos),
  sensitivity, and earlier epoch;
- subject probability equal to the mean utterance probability.

Train one isolated run per seed:

```bash
for seed in 123 1234 12345 123456 1234567; do
  python scripts/run_experiments.py \
    --data-root ./datasets/fixed_corrected_nooffset \
    --gpu 0 --seeds "$seed" --epochs 15 \
    --lr 1e-5 --batch-size 16 --weight-decay 1e-5 --dropout 0.1 \
    --ctc-enabled 0 --test-policy none \
    --run-tag "fixed_seed${seed}" \
    --output-dir "./runs/fixed/seed${seed}"
done
```

Training must use `--test-policy none`; no official-test loader is constructed.
After all choices are frozen, create a checkpoint index with these columns:

```text
variant,seed,checkpoint_path,data_root,learning_rate,batch_size,
weight_decay,dropout,ctc_enabled
```

Evaluate each frozen checkpoint once:

```bash
python scripts/eval_checkpoints.py \
  --checkpoint-index ./checkpoint_index.csv \
  --output-root ./results/fixed
```

The five official-test runs give:

- Macro-F1 `0.5764 ± 0.0361`;
- ROC-AUC `0.5281 ± 0.0536`.

These are mean ± sample SD across seeds at subject threshold 0.5, with no
ensemble.

## 4. PHQ-balanced 5-fold protocol

This protocol pools the official train, dev, and test subjects. It is internal
cross-validation over all 189 participants, not an independent official-test
evaluation.

Build one outer assignment stratified over PHQ-8 bins `0–4`, `5–9`, `10–14`,
`15–19`, and `20–24`. The published run uses the timestamp-offset variant:

```bash
python data_preparation/prepare_phq_stratified_train_test.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --offset --seed 123 \
  --out-root ./datasets/phq5
```

Expected total bin counts are `86, 46, 30, 20, 7`. For each bin, outer-fold
counts differ by at most one. The script also writes an audited inner-dev
partition, although the reported model trains on each fold's combined
train+dev subjects.

Run the exact ten-seed BCE-only matrix:

```bash
python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --output-root ./runs/phq5
```

The seeds are:

```text
1234, 12345, 123456, 1234567, 12345678,
2024, 2025, 2026, 2027, 2028
```

The runner fixes the relevant configuration:

| item | value |
|---|---|
| epochs | 15 |
| learning rate | `1e-4` |
| batch size | 16 |
| weight decay | `1e-5` |
| dropout | 0.3 |
| loss | BCE only |
| WavLM mask | true length |
| waveform normalization | valid audio first, then pad |
| train crop | deterministic epoch-keyed |
| optimizer | no-decay AdamW groups, warmup + cosine |
| evaluation views | head / center / tail (`multi3`) |
| subject aggregation | mean probability |
| threshold | 0.5 |
| ensemble | none |

The command evaluates held-out folds at all 15 epochs so the historical
post-hoc selection can be reproduced. The repository has one and only one
reported 5-fold convention: one shared epoch is selected across all ten seeds,
and that epoch is fixed to 14 in the summarizer.

```bash
python scripts/summarize_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --runs-root ./runs/phq5 \
  --output-root ./results/phq5
```

The result is:

- Macro-F1 `0.5537 ± 0.0225`;
- ROC-AUC `0.5412 ± 0.0177`.

Aggregation is performed in this order:

1. average utterance probabilities within subject;
2. concatenate the five held-out folds to obtain 189 OOF subjects per seed;
3. compute metrics once per seed;
4. report the mean and sample SD across ten seeds.

Epoch 14 was chosen using these same held-out trajectories. This is explicitly
post-hoc/test-tuned and `independent_test_performance=false`. Do not reinterpret
it as nested-CV or external-test performance.

## 5. Parallel execution

`run_phq_balanced_cv.py` is sequential by design. Partition work across GPUs
without changing the protocol:

```bash
python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 --output-root ./runs/phq5 \
  --gpu 0 --folds 0 --seeds 1234 12345
```

Every run writes to `fold_<FOLD>/seed<SEED>`. Distinct workers may safely target
the same output root when their fold/seed pairs do not overlap. Nonempty run
directories are never overwritten.

## 6. Verification

Run the lightweight unit/protocol suite:

```bash
python -m unittest discover -s tests -v
```

Hardware, CUDA, AMP, and library differences can prevent bit-exact
probabilities. Preserve the subject inventory, split hashes, selected epoch,
threshold, and aggregation order before comparing rounded metrics.
