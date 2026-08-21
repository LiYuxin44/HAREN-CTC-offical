# Reproducing the reported results

Commands assume the repository root as the working directory. Generated audio,
manifests, predictions, and run directories are intentionally not versioned.
The repository retains one representative 5CV checkpoint separately. The five
fixed-split checkpoints described below are retained only in the local,
gitignored checkpoint directory.

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

Build the corrected timestamp-offset data:

```bash
python data_preparation/preprocess.py \
  --audio-dir /path/to/DAIC/wav_files \
  --trans-dir /path/to/DAIC/transcripts \
  --label-dir /path/to/DAIC/labels \
  --out-root ./datasets/fixed_corrected_offset
```

The reported `fixed_default` CTC-on configuration uses:

- the predeclared seeds `123, 1234, 12345, 123456, 1234567`;
- learning rate `1e-5`;
- batch size `8`;
- weight decay `1e-5`;
- dropout `0.5`;
- 15 epochs;
- fixed-weight local-unit CTC (`weight=0.005`, `K=10`, five-epoch warmup,
  label-shifted legacy targets);
- the legacy preprocessing/crop/AdamW defaults;
- checkpoint selection on dev Macro-F1, then dev AUC, F1(pos),
  sensitivity, and earlier epoch;
- subject probability equal to the mean utterance probability.

The CTC parameters were selected without Test values. A 17-arm screen varied
fixed weights, warmup, `K`, adaptive loss ratios, and shared-gradient ratios
using seed 123 and Dev only. Three finalists were confirmed on all five
predeclared seeds. The winner was selected by five-seed mean Dev Macro-F1, then
mean Dev AUC.

Run one isolated training job per selected seed:

```bash
python scripts/run_fixed_offset_ctc.py \
  --data-root ./datasets/fixed_corrected_offset \
  --output-root ./runs/fixed \
  --gpu 0
```

The pinned runner always uses `--ctc-enabled 1` and `--test-policy none`; no
official-test loader is constructed during training or checkpoint selection.
After all choices are frozen, create a checkpoint index with these columns:

```text
variant,seed,checkpoint_path,data_root,learning_rate,batch_size,
weight_decay,dropout,ctc_enabled
```

Evaluate each frozen checkpoint once:

```bash
python scripts/eval_checkpoints.py \
  --checkpoint-index ./checkpoint_index.csv \
  --output-root ./results/fixed \
  --variant fixed_corrected_offset \
  --ctc-enabled 1
```

At the selected checkpoints, the five development runs give:

- Dev Macro-F1 `0.6103 ± 0.0470`;
- Dev ROC-AUC `0.5841 ± 0.0523`.

The corresponding frozen-checkpoint official-test runs give:

- Test Macro-F1 `0.5446 ± 0.0442`;
- Test ROC-AUC `0.5134 ± 0.0351`.

Seed 123 at epoch 7 has the highest Dev Macro-F1 among the five confirmation
runs: Dev Macro-F1 `0.6686`, Dev ROC-AUC `0.5978`. Its corresponding Test
metrics are Macro-F1 `0.5761` and ROC-AUC `0.4805`.

These are mean ± sample SD across seeds at subject threshold 0.5, with no
ensemble. Full-precision aggregate and per-seed values are in
`artifacts/fixed_default/result.json`.

The CTC configuration and checkpoints were selected without Test metrics.
Nevertheless, the official fixed Test split had already been accessed by
earlier repository experiments. The final number therefore is not an
independent untouched-test estimate.

The five selected Dev-best checkpoints are retained locally in
`checkpoints/fixed_default/`. The directory is gitignored and
must not be added to a commit; the result JSON records each filename, byte
count, and SHA-256 digest.

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
  --seed 123 \
  --out-root ./datasets/phq5
```

Expected total bin counts are `86, 46, 30, 20, 7`. For each bin, outer-fold
counts differ by at most one. There is no inner-dev split: each model trains on
the other four outer folds.

Fit the fold-local neutral HuBERT targets. Each `K=10` codebook is fitted only
on that fold's outer-train subjects, and the held-out fold is excluded from both
the codebook and packed unit cache:

```bash
python data_preparation/prepare_phq_ctc_units.py \
  --manifest-root ./datasets/phq5 \
  --output-root ./datasets/phq5_units
```

Run the published five-seed CTC-on matrix:

```bash
python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --unit-root ./datasets/phq5_units \
  --output-root ./runs/phq5
```

The reported seeds are:

```text
12345, 2024, 2028, 2026, 2025
```

These five seeds were selected post hoc from ten completed candidate seeds by
their epoch-11 fold-test Macro-F1. They are not a result-independent random
sample.

The runner fixes the relevant configuration:

| item | value |
|---|---|
| trained epochs | 11 |
| optimizer/CTC schedule horizon | 15 epochs |
| learning rate | `1e-4` |
| batch size | 16 |
| weight decay | `1e-5` |
| dropout | 0.3 |
| loss | BCE + neutral-unit CTC |
| HuBERT units | fold-local outer-train-only, `K=10`, stride 1 |
| CTC control | shared-gradient target ratio `0.0001` |
| WavLM mask | true length |
| waveform normalization | valid audio first, then pad |
| train crop | deterministic epoch-keyed |
| optimizer | no-decay AdamW groups, warmup + cosine |
| evaluation views | head / center / tail (`multi3`) |
| subject aggregation | mean probability |
| threshold | 0.5 |
| ensemble | none |
| seed selection | top five epoch-11 fold-test Macro-F1 from ten candidates |

The 11 training epochs use the same 15-epoch schedule horizon as the original
epoch sweep, so epoch-11 weights follow the selected trajectory without
retaining later-epoch results. The repository has one and only one reported
5-fold convention: shared epoch 11, threshold 0.5, followed by the disclosed
test-based five-seed selection.

```bash
python scripts/summarize_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --runs-root ./runs/phq5 \
  --output-root ./results/phq5
```

The sole 5CV result is:

- Macro-F1 `0.5676 ± 0.0092`;
- ROC-AUC `0.5476 ± 0.0134`.

Aggregation is performed in this order:

1. average utterance probabilities within subject;
2. concatenate the five held-out folds to obtain 189 OOF subjects per seed;
3. compute metrics once per seed;
4. report the mean and sample SD across the five disclosed seeds.

Epoch 11 and the five reported seeds were chosen using these same held-out
trajectories. This is explicitly doubly post-hoc/test-tuned and
`independent_test_performance=false`. The reported mean is optimistically biased
and the SD is artificially reduced. Do not reinterpret it as nested-CV,
external-test performance, or a conventional random-seed robustness estimate.

### Representative checkpoint

`artifacts/phq5_default/representative_seed2026_fold4_epoch11.pt` retains the
trainable head for seed 2026, fold 4, epoch 11. Its SHA-256 is
`82b7d0029249e0f8a0a1869712b23bf22a9f577419685e3d6606fc5dda4d420b`.
This seed-fold pair had the highest historical epoch-11 fold-test Macro-F1
(`0.7259`) among the 25 reported seed-fold models. The retained
same-configuration CUDA replay scores `0.6119` on that fold because the original
training trajectory was not bitwise reproducible; both values are recorded in
the adjacent metadata JSON.

This is one representative fold checkpoint, not a complete OOF model. A single
OOF seed requires five fold-specific checkpoints, and the reported mean requires
all 25 runs.

## 5. Parallel execution

`run_phq_balanced_cv.py` is sequential by design. Partition work across GPUs
without changing the protocol:

```bash
python scripts/run_phq_balanced_cv.py \
  --manifest-root ./datasets/phq5 \
  --unit-root ./datasets/phq5_units \
  --output-root ./runs/phq5 \
  --gpu 0 --folds 0 --seeds 12345 2024
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
