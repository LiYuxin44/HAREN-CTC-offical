"""
HAREN-CTC: WavLM-based speech depression detection with an auxiliary CTC head.

This single script defines the model AND runs the full multi-seed training /
evaluation loop. It is driven entirely by environment variables so the same file
reproduces every experiment reported in the paper (see docs/REPRODUCE.md).

Architecture
------------
  frozen WavLM-Large (24 layers, backbone weights + all stochastic ops frozen)
    -> AdaptiveWeightedPool      : learns a soft 2-group partition over the 24
                                   hidden-state layers (exponentially initialised)
    -> LearnablePositionalEmbedding + [CLS] token (per group)
    -> CoAttentionModule         : deep group attends over shallow group
    -> classifier head           : LayerNorm -> Linear(1024,64) -> SiLU -> Linear(64,1)
  Auxiliary CTC head             : Linear(1024, 2k+1) trained against online
                                   HuBERT k-means pseudo-phone targets (label-shifted
                                   so depressed/non-depressed use disjoint token sets).

Losses
------
  L = BCEWithLogits(cls) + effective_ctc_weight * CTCLoss.
  effective_ctc_weight is either a fixed warmup coefficient or a
  warmup-aware coefficient capped at a target CTC/BCE ratio.

Evaluation protocol
-------------------
  Utterance-level sigmoid probabilities are averaged at subject level.
  Metrics reported: macro-F1, F1_pos, F1_neg, sensitivity,
  specificity, ROC-AUC. Results are averaged over 5 seeds (mean +/- sd).

Environment variables (HPO and final runs pass these explicitly)
---------------------------------------------------------------
  SEEDS           comma list of seeds        (default 2029,123456,123,2032,12345678)
  NUM_EPOCHS      epochs per seed             (default 15)
  SCHEDULE_EPOCHS optimizer/CTC horizon       (default NUM_EPOCHS)
  BATCH_SIZE      batch size                  (default 8)
  LR              AdamW learning rate         (default 1e-5)
  DROPOUT         head dropout                 (default 0.5)
  WEIGHT_DECAY    AdamW weight decay           (default 1e-5)
  CTC_ENABLED     "1" enables auxiliary CTC    (default 1)
  CTC_WEIGHT      fixed/max CTC coefficient    (default 0.005)
  CTC_MODE        fixed | adaptive_ratio       (default fixed)
  CTC_TARGET_MODE label_shifted | neutral      (default label_shifted)
  CTC_TARGET_RATIO adaptive weighted-loss cap  (default 0.1)
  TEST_POLICY     none | final_only            (default none)
  DATA_ROOT       dir containing train/val/test (default datasets/fixed_corrected_offset)
  RUN_TAG         suffix appended to the log dir name

The WavLM path keeps Dataset-level 10-second zero-padding/full masks. The
HuBERT/CTC path separately tracks true sample lengths. Other invariants:
CTC_WARMUP_EPOCHS=5, num_groups=2, k=10 (21 CTC classes, blank=20).
"""
import os, random, logging, copy, wave, json, hashlib, time, contextlib
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, HubertModel
from global_unit_cache import (
    PackedUnitCache,
    file_sha256 as _cache_file_sha256,
    identifier_set_sha256,
)
from haren_ctc_utils import (
    aggregate_subject_predictions,
    build_ctc_targets,
    collapse_sliding_window_predictions,
    ctc_greedy_edit_counts,
    crop_waveform,
    deterministic_crop_start,
    effective_ctc_weight,
    EpochSeededWeightedSampler,
    evaluation_crop_start,
    feature_output_lengths,
    lengths_to_attention_mask,
    loss_gradient_diagnostics,
    masked_kmeans_batched,
    multi_view_crop_starts,
    named_parameter_hash,
    normalized_ctc_loss,
    normalized_frame_ce_loss,
    prepare_wavlm_inputs,
    routing_initial_logits,
    SharedGradientRatioController,
    sliding_window_starts,
    subject_balanced_sample_weights,
    temporal_logit_view,
    trainable_state_hash,
    waveform_quality_metrics,
)
from training_stability import (
    TrainableEMA,
    adamw_parameter_groups,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    warmup_cosine_multiplier,
)

# ───────────── 常量 / 路径 ─────────────
SEEDS           = [2029, 123456, 123, 2032, 12345678]
if os.environ.get('SEEDS'): SEEDS = [int(x) for x in os.environ['SEEDS'].split(',')]  # override for smoke
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS','15'))
SCHEDULE_EPOCHS = int(os.environ.get('SCHEDULE_EPOCHS', str(NUM_EPOCHS)))
BATCH_SIZE      = int(os.environ.get('BATCH_SIZE','8'))
LR              = float(os.environ.get('LR', '1e-5'))
DROPOUT         = float(os.environ.get('DROPOUT', '0.5'))
WEIGHT_DECAY    = float(os.environ.get('WEIGHT_DECAY', '1e-5'))
TRAINING_PROTOCOL = 'manuscript_final'
SPLIT_MODE      = os.environ.get('SPLIT_MODE', 'fixed').strip().lower()
FOLD_INDEX      = int(os.environ.get('FOLD_INDEX', '-1'))
NUM_WORKERS     = int(os.environ.get('NUM_WORKERS', '8'))
PREFETCH_FACTOR = int(os.environ.get('PREFETCH_FACTOR', '2'))
CTC_ENABLED     = os.environ.get('CTC_ENABLED', '1').strip()
CTC_MODE        = os.environ.get('CTC_MODE', 'fixed').strip().lower()
CTC_TARGET_MODE = os.environ.get(
    'CTC_TARGET_MODE', 'label_shifted'
).strip().lower()
TEST_POLICY     = os.environ.get('TEST_POLICY', 'none').strip().lower()
EVAL_CHECKPOINT = os.environ.get('EVAL_CHECKPOINT', '').strip()
CTC_WEIGHT = float(os.environ.get('CTC_WEIGHT', '0.005'))
CTC_WARMUP_EPOCHS = int(os.environ.get('CTC_WARMUP_EPOCHS', '5'))
CTC_TARGET_RATIO = float(os.environ.get('CTC_TARGET_RATIO', '0.1'))
CTC_K = int(os.environ.get('CTC_K', '10'))
CTC_GRAD_TARGET_RATIO = float(
    os.environ.get('CTC_GRAD_TARGET_RATIO', '0.1')
)
CTC_GRAD_UPDATE_INTERVAL = int(
    os.environ.get('CTC_GRAD_UPDATE_INTERVAL', '10')
)
CTC_GRAD_EMA_DECAY = float(os.environ.get('CTC_GRAD_EMA_DECAY', '0.9'))
CTC_WARMUP_RATIO = float(os.environ.get('CTC_WARMUP_RATIO', '0.1'))
CTC_LOSS_POLICY = os.environ.get(
    'CTC_LOSS_POLICY', 'legacy_mean'
).strip().lower()
TEMPORAL_HEAD_POLICY = os.environ.get(
    'TEMPORAL_HEAD_POLICY', 'legacy_2k1'
).strip().lower()
GLOBAL_UNIT_STRIDE = int(os.environ.get('GLOBAL_UNIT_STRIDE', '1'))
GLOBAL_UNIT_REVERSE = (
    os.environ.get('GLOBAL_UNIT_REVERSE', '0').strip() == '1'
)
CROP_ALIGNMENT_SAMPLES = int(
    os.environ.get('CROP_ALIGNMENT_SAMPLES', '1')
)
WAVLM_MASK_POLICY = os.environ.get(
    'WAVLM_MASK_POLICY', 'legacy_full'
).strip().lower()
ROUTING_INIT_POLICY = os.environ.get(
    'ROUTING_INIT_POLICY', 'legacy'
).strip().lower()
SAMPLING_POLICY = os.environ.get(
    'SAMPLING_POLICY', 'utterance_class_balanced'
).strip().lower()
AGGREGATION_MODE = os.environ.get(
    'AGGREGATION_POLICY', 'mean_probability'
).strip().lower()
TEMPORAL_TARGET_POLICY = os.environ.get(
    'TEMPORAL_TARGET_POLICY', 'local_kmeans_ctc'
).strip().lower()
GLOBAL_UNIT_CACHE = os.environ.get('GLOBAL_UNIT_CACHE', '').strip()
GRAD_DIAGNOSTIC_INTERVAL = int(
    os.environ.get('GRAD_DIAGNOSTIC_INTERVAL', '0')
)
GRAD_CLIP_NORM = float(os.environ.get('GRAD_CLIP_NORM', '0'))
GRAD_CLIP_POLICY = os.environ.get(
    'GRAD_CLIP_POLICY', 'global'
).strip().lower()
SILENCE_THRESHOLD_DBFS = float(
    os.environ.get('SILENCE_THRESHOLD_DBFS', '-40')
)
AMP_DTYPE_POLICY = os.environ.get('AMP_DTYPE', 'fp16').strip().lower()
EVAL_CROP_POLICY = os.environ.get(
    'EVAL_CROP_POLICY', 'head'
).strip().lower()
EVAL_WINDOW_STRIDE_SECONDS = float(
    os.environ.get('EVAL_WINDOW_STRIDE_SECONDS', '5')
)
WAVLM_PREPROCESS_POLICY = os.environ.get(
    'WAVLM_PREPROCESS_POLICY', 'legacy_prepad'
).strip().lower()
WAVLM_BATCH_PADDING_POLICY = os.environ.get(
    'WAVLM_BATCH_PADDING_POLICY', 'fixed_10s'
).strip().lower()
TRAIN_CROP_POLICY = os.environ.get(
    'TRAIN_CROP_POLICY', 'worker_random'
).strip().lower()
OPTIMIZER_POLICY = os.environ.get(
    'OPTIMIZER_POLICY', 'legacy_adamw'
).strip().lower()
HEAD_ARCH_POLICY = os.environ.get(
    'HEAD_ARCH_POLICY', 'legacy_17m'
).strip().lower()
HEAD_INIT_POLICY = os.environ.get(
    'HEAD_INIT_POLICY', 'legacy_stream'
).strip().lower()
LR_WARMUP_RATIO = float(os.environ.get('LR_WARMUP_RATIO', '0.1'))
LR_MIN_RATIO = float(os.environ.get('LR_MIN_RATIO', '0.1'))
EMA_DECAY = float(os.environ.get('EMA_DECAY', '0'))
SAVE_TRAINING_STATE = (
    os.environ.get('SAVE_TRAINING_STATE', '0').strip() == '1'
)
RESUME_CHECKPOINT = os.environ.get('RESUME_CHECKPOINT', '').strip()
TEMPORAL_EVAL_DIAGNOSTICS = (
    os.environ.get('TEMPORAL_EVAL_DIAGNOSTICS', '0').strip() == '1'
)

_DATA_ROOT = os.environ.get(
    'DATA_ROOT', './datasets/fixed_corrected_offset'
)
DIR_TRAIN = os.path.join(_DATA_ROOT, 'train')
DIR_VAL   = os.path.join(_DATA_ROOT, 'val')
DIR_TEST  = os.path.join(_DATA_ROOT, 'test')
TRAIN_MANIFEST = os.environ.get('TRAIN_MANIFEST', '').strip()
VAL_MANIFEST   = os.environ.get('VAL_MANIFEST', '').strip()

RUN_TAG        = os.environ.get('RUN_TAG', '').strip()

if CTC_ENABLED not in {'0', '1'}:
    raise ValueError(f"CTC_ENABLED must be '0' or '1', got {CTC_ENABLED!r}")
CTC_ENABLED = CTC_ENABLED == '1'
if NUM_EPOCHS <= 0 or SCHEDULE_EPOCHS < NUM_EPOCHS:
    raise ValueError(
        "NUM_EPOCHS must be positive and SCHEDULE_EPOCHS must be at least "
        "NUM_EPOCHS"
    )
if CTC_MODE not in {'fixed', 'adaptive_ratio', 'shared_grad_norm'}:
    raise ValueError(f"Unsupported CTC_MODE: {CTC_MODE!r}")
if CTC_TARGET_MODE not in {'label_shifted', 'neutral'}:
    raise ValueError(f"Unsupported CTC_TARGET_MODE: {CTC_TARGET_MODE!r}")
if (
    CTC_WEIGHT < 0
    or CTC_WARMUP_EPOCHS < 0
    or CTC_TARGET_RATIO <= 0
    or CTC_K <= 0
    or CTC_GRAD_TARGET_RATIO <= 0
    or CTC_GRAD_UPDATE_INTERVAL <= 0
    or not 0 <= CTC_GRAD_EMA_DECAY < 1
    or not 0 <= CTC_WARMUP_RATIO < 1
    or GLOBAL_UNIT_STRIDE <= 0
    or CROP_ALIGNMENT_SAMPLES <= 0
):
    raise ValueError("Invalid CTC weight, warmup, or target ratio")
if CTC_ENABLED and CTC_WEIGHT <= 0:
    raise ValueError("CTC_ENABLED requires a positive CTC_WEIGHT")
if CTC_LOSS_POLICY not in {'legacy_mean', 'normalized_fp32'}:
    raise ValueError(f"Unsupported CTC_LOSS_POLICY: {CTC_LOSS_POLICY!r}")
if CTC_LOSS_POLICY == 'normalized_fp32' and CTC_TARGET_MODE != 'neutral':
    raise ValueError("normalized_fp32 CTC only supports neutral targets")
if TEMPORAL_HEAD_POLICY not in {'legacy_2k1', 'neutral_k1'}:
    raise ValueError(
        f"Unsupported TEMPORAL_HEAD_POLICY: {TEMPORAL_HEAD_POLICY!r}"
    )
if TEMPORAL_HEAD_POLICY == 'neutral_k1' and CTC_TARGET_MODE != 'neutral':
    raise ValueError("neutral_k1 temporal head requires neutral CTC targets")
if WAVLM_MASK_POLICY not in {'legacy_full', 'true_length'}:
    raise ValueError(f"Unsupported WAVLM_MASK_POLICY: {WAVLM_MASK_POLICY!r}")
if ROUTING_INIT_POLICY not in {'legacy', 'probability_correct'}:
    raise ValueError(
        f"Unsupported ROUTING_INIT_POLICY: {ROUTING_INIT_POLICY!r}"
    )
if SAMPLING_POLICY not in {
    'utterance_class_balanced', 'subject_class_balanced'
}:
    raise ValueError(f"Unsupported SAMPLING_POLICY: {SAMPLING_POLICY!r}")
if AGGREGATION_MODE not in {
    'mean_probability',
    'confidence_weighted_probability',
    'confidence_weighted_vote',
}:
    raise ValueError(f"Unsupported AGGREGATION_POLICY: {AGGREGATION_MODE!r}")
if TEMPORAL_TARGET_POLICY not in {
    'local_kmeans_ctc', 'global_units_ctc', 'global_units_frame_ce'
}:
    raise ValueError(
        f"Unsupported TEMPORAL_TARGET_POLICY: {TEMPORAL_TARGET_POLICY!r}"
    )
if (
    CTC_ENABLED
    and TEMPORAL_TARGET_POLICY.startswith('global_units_')
    and not GLOBAL_UNIT_CACHE
):
    raise ValueError(
        "GLOBAL_UNIT_CACHE is required for global temporal target policies"
    )
if GRAD_DIAGNOSTIC_INTERVAL < 0 or GRAD_CLIP_NORM < 0:
    raise ValueError("Gradient diagnostic interval and clip norm must be nonnegative")
if GRAD_CLIP_POLICY not in {'global', 'task_grouped'}:
    raise ValueError(f"Unsupported GRAD_CLIP_POLICY: {GRAD_CLIP_POLICY!r}")
if AMP_DTYPE_POLICY not in {'fp16', 'bf16', 'fp32'}:
    raise ValueError(f"Unsupported AMP_DTYPE: {AMP_DTYPE_POLICY!r}")
if EVAL_CROP_POLICY not in {'head', 'center', 'tail', 'multi3', 'sliding_all'}:
    raise ValueError(f"Unsupported EVAL_CROP_POLICY: {EVAL_CROP_POLICY!r}")
if not 0 < EVAL_WINDOW_STRIDE_SECONDS <= 10:
    raise ValueError("EVAL_WINDOW_STRIDE_SECONDS must be in (0, 10]")
if EVAL_CROP_POLICY == 'sliding_all' and SPLIT_MODE != 'eval_only':
    raise ValueError("sliding_all is restricted to frozen eval_only runs")
if WAVLM_PREPROCESS_POLICY not in {'legacy_prepad', 'valid_then_pad'}:
    raise ValueError(
        f"Unsupported WAVLM_PREPROCESS_POLICY: {WAVLM_PREPROCESS_POLICY!r}"
    )
if WAVLM_BATCH_PADDING_POLICY not in {'fixed_10s', 'longest'}:
    raise ValueError(
        "WAVLM_BATCH_PADDING_POLICY must be fixed_10s or longest"
    )
if (
    WAVLM_PREPROCESS_POLICY == 'legacy_prepad'
    and WAVLM_BATCH_PADDING_POLICY != 'fixed_10s'
):
    raise ValueError("legacy_prepad requires fixed_10s batch padding")
if TRAIN_CROP_POLICY not in {'worker_random', 'epoch_keyed'}:
    raise ValueError(
        f"Unsupported TRAIN_CROP_POLICY: {TRAIN_CROP_POLICY!r}"
    )
if OPTIMIZER_POLICY not in {
    'legacy_adamw', 'no_decay_warmup_cosine'
}:
    raise ValueError(f"Unsupported OPTIMIZER_POLICY: {OPTIMIZER_POLICY!r}")
if HEAD_ARCH_POLICY not in {'legacy_17m', 'compact_9m'}:
    raise ValueError(f"Unsupported HEAD_ARCH_POLICY: {HEAD_ARCH_POLICY!r}")
if HEAD_INIT_POLICY not in {'legacy_stream', 'component_seeded_v1'}:
    raise ValueError(f"Unsupported HEAD_INIT_POLICY: {HEAD_INIT_POLICY!r}")
if (
    not 0.0 <= LR_WARMUP_RATIO < 1.0
    or not 0.0 <= LR_MIN_RATIO <= 1.0
    or not 0.0 <= EMA_DECAY < 1.0
):
    raise ValueError("Invalid LR schedule or EMA configuration")
if RESUME_CHECKPOINT and (
    len(SEEDS) != 1 or TRAIN_CROP_POLICY != 'epoch_keyed'
):
    raise ValueError(
        "Exact resume requires one seed and TRAIN_CROP_POLICY=epoch_keyed"
    )
SAMPLER_NAME = (
    (
        'epoch_seeded_weighted_random'
        if SAMPLING_POLICY == 'utterance_class_balanced'
        else 'epoch_seeded_subject_class_balanced'
    )
    if TRAIN_CROP_POLICY == 'epoch_keyed'
    else (
        'weighted_random'
        if SAMPLING_POLICY == 'utterance_class_balanced'
        else 'subject_class_balanced'
    )
)
if TEST_POLICY not in {'none', 'final_only'}:
    raise ValueError(
        f"TEST_POLICY must be 'none' or 'final_only', got {TEST_POLICY!r}"
    )
if SPLIT_MODE == 'eval_only':
    if (
        not EVAL_CHECKPOINT
        or TEST_POLICY != 'none'
        or TRAIN_MANIFEST
        or not VAL_MANIFEST
    ):
        raise ValueError(
            "eval_only requires EVAL_CHECKPOINT, VAL_MANIFEST, "
            "TEST_POLICY=none, and no TRAIN_MANIFEST"
        )
    if TEMPORAL_EVAL_DIAGNOSTICS and (
        not CTC_ENABLED
        or TEMPORAL_TARGET_POLICY != 'global_units_ctc'
        or GLOBAL_UNIT_REVERSE
        or EVAL_CROP_POLICY != 'head'
    ):
        raise ValueError(
            "Temporal eval diagnostics require forward global-unit CTC"
        )
elif (TEST_POLICY == 'final_only') != bool(EVAL_CHECKPOINT):
    raise ValueError(
        "TEST_POLICY=final_only requires EVAL_CHECKPOINT, and EVAL_CHECKPOINT "
        "is forbidden when TEST_POLICY=none"
    )
elif TEMPORAL_EVAL_DIAGNOSTICS:
    raise ValueError("Temporal eval diagnostics require eval_only mode")
if SPLIT_MODE not in {
    'fixed', 'cv', 'inner', 'train_only', 'eval_only', 'test_tune'
}:
    raise ValueError(
        "SPLIT_MODE must be fixed, cv, inner, train_only, eval_only, "
        "or test_tune; "
        f"got {SPLIT_MODE!r}"
    )
if SPLIT_MODE in {'cv', 'inner', 'test_tune'} and (
    not TRAIN_MANIFEST or not VAL_MANIFEST
):
    raise ValueError(
        f"{SPLIT_MODE} mode requires TRAIN_MANIFEST and VAL_MANIFEST"
    )
if SPLIT_MODE == 'train_only' and (
    not TRAIN_MANIFEST or VAL_MANIFEST
):
    raise ValueError(
        "train_only mode requires TRAIN_MANIFEST and forbids VAL_MANIFEST"
    )
if (
    SPLIT_MODE in {'cv', 'inner', 'eval_only', 'test_tune'}
    and FOLD_INDEX not in range(5)
):
    raise ValueError(
        f"{SPLIT_MODE} mode requires FOLD_INDEX in 0..4, got {FOLD_INDEX}"
    )
if (
    SPLIT_MODE in {'cv', 'inner', 'train_only', 'eval_only', 'test_tune'}
    and TEST_POLICY != 'none'
):
    raise ValueError(
        f"{SPLIT_MODE} mode does not expose the official test split"
    )

DEFAULT_LOG_DIR = (
    f"logs_{SPLIT_MODE}_{TRAINING_PROTOCOL}_CTC"
    f"{str(CTC_WEIGHT).replace('.', 'p')}"
    + (('_' + RUN_TAG) if RUN_TAG else '')
)
LOG_DIR       = os.environ.get('OUTPUT_DIR', '').strip() or DEFAULT_LOG_DIR
LOG_BASE_NAME = 'log_fixed_split'
PER_EPOCH_CSV    = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_per_epoch.csv')
PER_RUN_CSV       = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_per_run.csv')
SUMMARY_CSV      = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_summary.csv')
REPORT_DIR       = os.path.join(LOG_DIR, 'reports')
CHECKPOINT_DIR   = os.path.join(LOG_DIR, 'checkpoints')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
hubert_feature_extractor = (
    AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
    if (
        CTC_ENABLED
        and TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
        and not EVAL_CHECKPOINT
    )
    else None
)

# ───────────── PHQ严重程度分类函数（AudioDataset 在 metadata 里仍会用到） ─────────────
def phq_to_severity(phq_score: int) -> int:
    if phq_score <= 9:
        return 0  # normal + mild (0-9)
    elif phq_score <= 14:
        return 1  # moderate (10-14)
    else:
        return 2  # severe + very severe (15-24)

severity_labels = ['Normal+Mild(0-9)', 'Moderate(10-14)', 'Severe+VerySevere(15-24)']

# ───────────── logging 配置 ─────────────
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
main_log_file = os.path.join(LOG_DIR, f"{LOG_BASE_NAME}_main.log")

logging.basicConfig(
    filename=main_log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.root.addHandler(console_handler)

# ───────────── Seed 设置 ─────────────
def set_seed(seed=51, seed_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ───────────── 指标函数 ─────────────
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn+fp) else 0.

def metrics(y_true, y_pred, y_probs=None):
    result = dict(
        f1_pos   = f1_score(y_true, y_pred, pos_label=1),
        f1_neg   = f1_score(y_true, y_pred, pos_label=0),
        f1_macro = f1_score(y_true, y_pred, average='macro'),
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0),
        sens     = recall_score(y_true, y_pred),
        spec     = specificity(y_true, y_pred),
        cm       = confusion_matrix(y_true, y_pred, labels=[0,1]),
    )
    if y_probs is not None:
        try:
            result['auc'] = roc_auc_score(y_true, y_probs)
        except ValueError:
            result['auc'] = 0.0
    return result

def log_metrics(tag, m):
    cm   = m.get("cm")
    msg  = "  ".join(
        f"{k}:{v:.4f}"
        for k, v in m.items()
        if k != "cm" and isinstance(v, (int, float))
    )
    logging.info(f"[{tag}] {msg}")
    if cm is not None:
        logging.info(f"[{tag}] Confusion\n{cm}")

# ───────────── 构建 wav 索引 ─────────────
def build_wav_index(dirs):
    """
    扫描多个目录，把所有满足:
      - *.wav
      - 存在 .label 和 .phq_label
    的样本加入列表。
    """
    wavs = []
    for root in dirs:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            if not fn.endswith('.wav'):
                continue
            wav_path = os.path.join(root, fn)
            binary_lab = wav_path.replace('.wav', '.label')
            phq_lab    = wav_path.replace('.wav', '.phq_label')
            if os.path.exists(binary_lab) and os.path.exists(phq_lab):
                wavs.append(wav_path)
    return sorted(wavs)


def build_manifest_index(manifest_path):
    """Load wav paths from a fold manifest; relative paths resolve via DATA_ROOT."""
    frame = pd.read_csv(manifest_path)
    if 'path' not in frame.columns:
        raise ValueError(f"Manifest lacks a 'path' column: {manifest_path}")
    wavs = []
    for value in frame['path'].astype(str):
        wav_path = value if os.path.isabs(value) else os.path.join(_DATA_ROOT, value)
        binary_lab = wav_path.replace('.wav', '.label')
        phq_lab = wav_path.replace('.wav', '.phq_label')
        if not (os.path.exists(wav_path) and os.path.exists(binary_lab) and os.path.exists(phq_lab)):
            raise FileNotFoundError(f"Incomplete manifest sample: {wav_path}")
        wavs.append(os.path.normpath(wav_path))
    if len(wavs) != len(set(wavs)):
        raise ValueError(f"Duplicate wav paths in manifest: {manifest_path}")
    return wavs


if SPLIT_MODE == 'eval_only':
    TRAIN_WAVS = []
    VAL_WAVS = []
    TEST_WAVS = build_manifest_index(VAL_MANIFEST)
elif SPLIT_MODE in {'cv', 'inner', 'train_only', 'test_tune'}:
    TRAIN_WAVS = build_manifest_index(TRAIN_MANIFEST)
    VAL_WAVS = (
        [] if SPLIT_MODE == 'train_only'
        else build_manifest_index(VAL_MANIFEST)
    )
    TEST_WAVS = []
else:
    TRAIN_WAVS = build_wav_index([DIR_TRAIN])
    VAL_WAVS = build_wav_index([DIR_VAL])
    TEST_WAVS = (
        build_wav_index([DIR_TEST]) if TEST_POLICY == 'final_only' else []
    )

if (
    (not EVAL_CHECKPOINT and len(TRAIN_WAVS) == 0)
    or (
        not EVAL_CHECKPOINT
        and SPLIT_MODE != 'train_only'
        and len(VAL_WAVS) == 0
    )
    or (
        (TEST_POLICY == 'final_only' or SPLIT_MODE == 'eval_only')
        and len(TEST_WAVS) == 0
    )
):
    raise RuntimeError(
        f"Empty split: train={len(TRAIN_WAVS)}, val={len(VAL_WAVS)}, test={len(TEST_WAVS)}. "
        f"Check DATA_ROOT and split manifests."
    )
# 跨 split subject leakage 检查（一次性，开训前）
def _subjs(wavs):
    return set(os.path.basename(p).split('_')[0] for p in wavs)
train_subj_set, val_subj_set, test_subj_set = _subjs(TRAIN_WAVS), _subjs(VAL_WAVS), _subjs(TEST_WAVS)
split_pairs = (
    [('train', train_subj_set, 'val', val_subj_set)]
    if not EVAL_CHECKPOINT
    else []
)
if TEST_WAVS:
    split_pairs.extend([
        ('train', train_subj_set, 'test', test_subj_set),
        ('val', val_subj_set, 'test', test_subj_set),
    ])
for a_name, a, b_name, b in split_pairs:
    inter = a & b
    if inter:
        raise RuntimeError(f"Subject leakage between {a_name} and {b_name}: {sorted(inter)[:20]}")
logging.info(
    "No subject overlap across active splits. mode=%s protocol=%s fold=%d | "
    "subjects train/val/test=%d/%d/%d | utt=%d/%d/%d",
    SPLIT_MODE, TRAINING_PROTOCOL, FOLD_INDEX,
    len(train_subj_set), len(val_subj_set), len(test_subj_set),
    len(TRAIN_WAVS), len(VAL_WAVS), len(TEST_WAVS),
)

# ───────────── Dataset 定义（与 5cv 版本完全一致） ─────────────
class AudioDataset(Dataset):
    def __init__(self, wav_paths, feat_extractor, label_encoder,
                 subject_filter=None, training=True, max_sec=10.0):
        self.feat, self.le = feat_extractor, label_encoder
        self.training, self.max_sec  = training, max_sec
        self._subject_filter = set(map(str, subject_filter)) if subject_filter is not None else None

        self.info = []
        for wav in wav_paths:
            fn = os.path.basename(wav)
            subj = fn.split('_')[0]
            if self._subject_filter is not None and subj not in self._subject_filter:
                continue

            binary_lab = wav.replace('.wav', '.label')
            phq_lab    = wav.replace('.wav', '.phq_label')
            if not (os.path.exists(binary_lab) and os.path.exists(phq_lab)):
                continue

            with open(binary_lab) as f:
                binary_label = int(f.read().strip())
            with open(phq_lab) as f:
                phq_score = int(f.read().strip())

            with wave.open(wav, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            if sample_rate != 16000 or channels != 1 or sample_width != 2:
                raise ValueError(
                    f"Expected mono 16 kHz PCM16 wav, got sr={sample_rate}, "
                    f"channels={channels}, width={sample_width}: {wav}"
                )
            base_info = dict(
                path=wav,
                binary_label=binary_label,
                phq_score=phq_score,
                phq_severity=phq_to_severity(phq_score),
                sr=sample_rate,
                nfrm=num_frames,
                subject=subj,
            )
            if not self.training and EVAL_CROP_POLICY in {
                'multi3',
                'sliding_all',
            }:
                segment_samples = int(self.max_sec * sample_rate)
                if EVAL_CROP_POLICY == 'multi3':
                    starts = multi_view_crop_starts(
                        num_frames,
                        segment_samples,
                        alignment=CROP_ALIGNMENT_SAMPLES,
                    )
                else:
                    stride_samples = int(
                        EVAL_WINDOW_STRIDE_SECONDS * sample_rate
                    )
                    starts = sliding_window_starts(
                        num_frames, segment_samples, stride_samples
                    )
                for window_index, start in enumerate(starts):
                    stem, extension = os.path.splitext(wav)
                    self.info.append(
                        {
                            **base_info,
                            'forced_crop_start': int(start),
                            'prediction_path': (
                                f"{stem}__window{window_index:04d}{extension}"
                            ),
                        }
                    )
            else:
                self.info.append(
                    {
                        **base_info,
                        'forced_crop_start': None,
                        'prediction_path': wav,
                    }
                )

        if len(self.info) == 0:
            raise RuntimeError("AudioDataset has 0 samples after filtering. "
                               "Check subject_filter and pool indexing.")

        if self.le:
            ys = self.le.transform([x['binary_label'] for x in self.info])
            for d, y in zip(self.info, ys):
                d['label'] = int(y)
        else:
            for d in self.info:
                d['label'] = d['binary_label']

    @property
    def labels(self):
        return [x['label'] for x in self.info]

    @property
    def subjects(self):
        return [x['subject'] for x in self.info]

    def __len__(self):
        return len(self.info)

    def _crop(
        self,
        wav,
        total,
        sr,
        *,
        identifier,
        crop_epoch=None,
        crop_draw=None,
    ):
        seg = int(self.max_sec * sr)
        if self.training and total > seg:
            if TRAIN_CROP_POLICY == 'epoch_keyed':
                if crop_epoch is None or crop_draw is None:
                    raise RuntimeError(
                        "epoch_keyed crop requires sampler epoch/draw keys"
                    )
                st = deterministic_crop_start(
                    identifier,
                    total,
                    seg,
                    seed=SEED,
                    epoch=int(crop_epoch),
                    draw=int(crop_draw),
                    alignment=CROP_ALIGNMENT_SAMPLES,
                )
            else:
                choices = (total - seg) // CROP_ALIGNMENT_SAMPLES + 1
                st = random.randrange(choices) * CROP_ALIGNMENT_SAMPLES
            cropped = crop_waveform(wav, seg, st)
            valid_samples = int(cropped.size(-1))
            return cropped, valid_samples, st
        if total <= seg:
            cropped = crop_waveform(wav, seg, 0)
            valid_samples = int(cropped.size(-1))
            return cropped, valid_samples, 0
        eval_start = evaluation_crop_start(
            total,
            seg,
            EVAL_CROP_POLICY,
            alignment=CROP_ALIGNMENT_SAMPLES,
        )
        cropped = crop_waveform(wav, seg, eval_start)
        valid_samples = int(cropped.size(-1))
        return cropped, valid_samples, eval_start

    def __getitem__(self, idx):
        crop_epoch = crop_draw = None
        if isinstance(idx, (tuple, list)):
            if len(idx) != 3:
                raise ValueError("Expected (sample_index, epoch, draw)")
            idx, crop_epoch, crop_draw = map(int, idx)
        d = self.info[idx]
        with wave.open(d['path'], 'rb') as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
        wav = torch.from_numpy(samples).unsqueeze(0)
        if d['forced_crop_start'] is not None:
            segment_samples = int(self.max_sec * d['sr'])
            crop_start = int(d['forced_crop_start'])
            wav = crop_waveform(wav, segment_samples, crop_start)
            valid_samples = int(wav.size(-1))
        else:
            wav, valid_samples, crop_start = self._crop(
                wav,
                d['nfrm'],
                d['sr'],
                identifier=d['path'],
                crop_epoch=crop_epoch,
                crop_draw=crop_draw,
            )
        return (
            wav.squeeze(0),
            valid_samples,
            crop_start,
            d['label'],
            d['phq_score'],
            d['phq_severity'],
            d['prediction_path'],
            d['subject'],
        )


def featurize_waveforms(waves, valid_samples, extractor):
    """Featurize cropped audio under explicit normalize/pad/mask policies."""
    return prepare_wavlm_inputs(
        waves,
        valid_samples,
        extractor,
        preprocess_policy=WAVLM_PREPROCESS_POLICY,
        mask_policy=WAVLM_MASK_POLICY,
        padding_policy=WAVLM_BATCH_PADDING_POLICY,
        max_samples=160000,
    )


def collate(batch):
    (
        waves,
        valid_samples,
        _crop_starts,
        labels,
        phq_scores,
        phq_severities,
        paths,
        subjects,
    ) = zip(*batch)
    x, attn_mask = featurize_waveforms(
        waves, valid_samples, feature_extractor
    )
    return x, attn_mask, torch.tensor(labels), torch.tensor(phq_scores), torch.tensor(phq_severities), list(paths), list(subjects)


def collate_with_waves(batch):
    (
        waves,
        valid_samples,
        crop_starts,
        labels,
        phq_scores,
        phq_severities,
        paths,
        subjects,
    ) = zip(*batch)
    x, attn_mask = featurize_waveforms(
        waves, valid_samples, feature_extractor
    )
    ctc_waves = [
        wave[: int(length)] for wave, length in zip(waves, valid_samples)
    ]
    return (
        x,
        attn_mask,
        torch.tensor(labels),
        torch.tensor(phq_scores),
        torch.tensor(phq_severities),
        list(paths),
        list(subjects),
        ctc_waves,
        list(crop_starts),
    )

# ───────────── 模型定义（与 5cv 版本完全一致） ─────────────
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        if not pre_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        if self.pre_norm:
            x = self.norm(x)
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return residual + x
        else:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = residual + x
            x = self.norm(x)
            return x

class CoAttentionModule(nn.Module):
    def __init__(
        self,
        d_model_size,
        num_heads=2,
        dropout=0.3,
        d_ff=None,
        pre_norm=True,
        external_qkv=True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model_size

        self.query = (
            nn.Linear(d_model_size, d_model_size)
            if external_qkv
            else nn.Identity()
        )
        self.key = (
            nn.Linear(d_model_size, d_model_size)
            if external_qkv
            else nn.Identity()
        )
        self.value = (
            nn.Linear(d_model_size, d_model_size)
            if external_qkv
            else nn.Identity()
        )
        self.attention = nn.MultiheadAttention(d_model_size, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model_size)
        self.norm2 = nn.LayerNorm(d_model_size)
        self.ffn = FFN(d_model_size, d_ff, dropout=dropout, pre_norm=pre_norm)

    def forward(self, shallow_output, deep_output, key_padding_mask=None):
        norm_shallow = self.norm1(shallow_output)
        norm_deep = self.norm2(deep_output)
        q = self.query(norm_deep).permute(1, 0, 2)
        k = self.key(norm_shallow).permute(1, 0, 2)
        v = self.value(norm_shallow).permute(1, 0, 2)
        attn_output, _ = self.attention(q, k, v, key_padding_mask=key_padding_mask)
        attn_output = attn_output.permute(1, 0, 2)
        output = deep_output + self.dropout(attn_output)
        output = self.ffn(output)
        return output

class AdaptiveWeightedPool(nn.Module):
    def __init__(self, num_layers: int = 24, hidden_size: int = 1024,
                 num_groups: int = 2, init_std: float = 0.02,
                 init_strategy: str = "exponential", alpha: float = 0.95,
                 init_policy: str = "legacy"):
        super().__init__()
        self.num_layers = num_layers
        self.selected_layers = list(range(num_layers))
        self.num_selected_layers = len(self.selected_layers)
        self.num_groups = num_groups

        self.group_assignment = nn.Parameter(
            torch.normal(mean=0.0, std=init_std, size=(self.num_selected_layers, num_groups))
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.initialize_exponential(alpha=alpha, policy=init_policy)

    def forward(self, hidden_states: list):
        selected_states = [hidden_states[i] for i in self.selected_layers]
        stacked_states = torch.stack(selected_states, dim=0)
        num_layers, batch_size, seq_len, hidden_size = stacked_states.shape
        stacked_states = self.layer_norm(stacked_states.view(-1, hidden_size))
        stacked_states = stacked_states.view(num_layers, batch_size, seq_len, hidden_size)
        group_probs = F.softmax(self.group_assignment, dim=-1)
        group_outputs = torch.einsum('lbsh,lg->bshg', stacked_states, group_probs)
        group_outputs_list = [group_outputs[..., g] for g in range(self.num_groups)]
        return group_outputs_list, group_probs, group_probs

    def initialize_exponential(self, alpha=0.9, policy="legacy"):
        with torch.no_grad():
            self.group_assignment.copy_(
                routing_initial_logits(
                    self.num_selected_layers,
                    alpha=alpha,
                    policy=policy,
                )
            )

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        t = x.size(1)
        x = x + self.pe[:, :t]
        return self.dropout(x)

class WavLMClassificationModel(nn.Module):
    def __init__(
        self,
        wavlm_model,
        num_labels=1,
        num_groups=2,
        dropout=0.3,
        ctc_weight=0.005,
        k=10,
        routing_init_policy="legacy",
        head_arch_policy="legacy_17m",
        head_init_policy="legacy_stream",
        temporal_head_policy="legacy_2k1",
        initialization_seed=None,
    ):
        super().__init__()
        self.wavlm_model = wavlm_model
        # Fully freeze backbone behavior: no grad, no SpecAugment/dropout/layerdrop.
        self._freeze_wavlm_backbone()
        self.k = k
        self.ctc_weight = ctc_weight
        self.head_arch_policy = head_arch_policy
        self.head_init_policy = head_init_policy
        self.temporal_head_policy = temporal_head_policy
        d_model_size = self.wavlm_model.config.hidden_size
        if head_arch_policy not in {"legacy_17m", "compact_9m"}:
            raise ValueError(f"Unsupported head architecture: {head_arch_policy}")
        if head_init_policy not in {"legacy_stream", "component_seeded_v1"}:
            raise ValueError(f"Unsupported head initialization: {head_init_policy}")
        if temporal_head_policy not in {"legacy_2k1", "neutral_k1"}:
            raise ValueError(
                f"Unsupported temporal head policy: {temporal_head_policy}"
            )
        if head_init_policy == "component_seeded_v1" and initialization_seed is None:
            raise ValueError("component_seeded_v1 requires initialization_seed")
        compact = head_arch_policy == "compact_9m"
        # Always consume the historical K=10 head RNG before shared modules.
        # This preserves K=10 initialization and keeps the shared trunk paired
        # when a larger global codebook changes only the temporal head shape.
        historical_ctc_classifier = nn.Linear(d_model_size, 21)

        def build_temporal_head():
            if temporal_head_policy == "neutral_k1":
                return nn.Sequential(
                    nn.LayerNorm(d_model_size),
                    nn.Linear(d_model_size, int(k) + 1),
                )
            return (
                historical_ctc_classifier
                if int(k) == 10
                else nn.Linear(d_model_size, 2 * int(k) + 1)
            )

        self.adaptive_pool = AdaptiveWeightedPool(
            num_layers=24, hidden_size=1024, num_groups=num_groups,
            init_std=0.02, init_strategy="exponential", alpha=0.95,
            init_policy=routing_init_policy,
        )
        self.positional_encoding = LearnablePositionalEmbedding(
            d_model_size,
            dropout=dropout,
            max_len=512 if compact else 2000,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model_size))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.co_attention_module = CoAttentionModule(
            d_model_size,
            num_heads=2,
            dropout=dropout,
            d_ff=(2 if compact else 4) * d_model_size,
            external_qkv=not compact,
        )
        self.pre_class_norm = nn.LayerNorm(d_model_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model_size),
            nn.Linear(d_model_size, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels)
        )
        # Init ONLY newly added heads — never re-init pretrained WavLM.
        if head_init_policy == "legacy_stream":
            historical_ctc_classifier.apply(self._init_weights)
            for m in (
                self.adaptive_pool, self.positional_encoding,
                self.co_attention_module, self.pre_class_norm, self.classifier,
            ):
                m.apply(self._init_weights)
            self.ctc_classifier = build_temporal_head()
            if (
                temporal_head_policy == "neutral_k1"
                or int(k) != 10
            ):
                self.ctc_classifier.apply(self._init_weights)
        else:
            self.ctc_classifier = build_temporal_head()
            self._initialize_component_seeded(int(initialization_seed))

    def _freeze_wavlm_backbone(self):
        """Lock WavLM weights AND train-time stochastic ops."""
        cfg = self.wavlm_model.config
        for key in (
            'mask_time_prob', 'mask_time_length', 'mask_feature_prob', 'mask_feature_length',
            'hidden_dropout', 'attention_dropout', 'activation_dropout', 'feat_proj_dropout',
            'final_dropout', 'layerdrop',
        ):
            if hasattr(cfg, key):
                setattr(cfg, key, 0.0)
        for p in self.wavlm_model.parameters():
            p.requires_grad = False
        self.wavlm_model.eval()

    def train(self, mode: bool = True):
        # Keep WavLM in eval even when the classification head is training.
        super().train(mode)
        self.wavlm_model.eval()
        return self

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _component_generator(seed, name):
        material = hashlib.sha256(
            f"{int(seed)}\0{name}".encode("utf-8")
        ).digest()
        value = int.from_bytes(material[:8], byteorder="big") % (2**63 - 1)
        return torch.Generator(device="cpu").manual_seed(value)

    def _initialize_component_seeded(self, seed):
        roots = {
            "adaptive_pool": self.adaptive_pool,
            "co_attention_module": self.co_attention_module,
            "pre_class_norm": self.pre_class_norm,
            "classifier": self.classifier,
            "ctc_classifier": self.ctc_classifier,
        }
        for root_name, root in roots.items():
            for child_name, module in root.named_modules():
                name = (
                    root_name
                    if not child_name
                    else f"{root_name}.{child_name}"
                )
                if isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(
                        module.in_proj_weight,
                        generator=self._component_generator(
                            seed, f"{name}.in_proj_weight"
                        ),
                    )
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(
                        module.weight,
                        generator=self._component_generator(
                            seed, f"{name}.weight"
                        ),
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.normal_(
            self.positional_encoding.pe,
            mean=0.0,
            std=0.02,
            generator=self._component_generator(
                seed, "positional_encoding.pe"
            ),
        )
        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02,
            generator=self._component_generator(seed, "cls_token"),
        )

    def _downsampled_mask(self, attention_mask, feature_len):
        if attention_mask is None:
            return None
        with torch.no_grad():
            feat_lengths = self.ctc_input_lengths(attention_mask).clamp_max(
                feature_len
            )
            rng = torch.arange(feature_len, device=attention_mask.device).unsqueeze(0)
            key_padding_mask = rng >= feat_lengths.unsqueeze(1)
            return key_padding_mask

    def ctc_input_lengths(self, attention_mask):
        """Return valid WavLM feature frames for every waveform."""
        if attention_mask is None:
            raise ValueError("attention_mask is required for per-sample CTC lengths")
        return self.ctc_input_lengths_from_samples(
            attention_mask.sum(dim=-1)
        )

    def ctc_input_lengths_from_samples(self, sample_lengths):
        """Return convolutional frame lengths for true, pre-padding audio."""
        return feature_output_lengths(
            sample_lengths,
            self.wavlm_model.config.conv_kernel,
            self.wavlm_model.config.conv_stride,
        )

    def forward(self, input_values, attention_mask=None):
        # Deterministic frozen features (no autograd through WavLM).
        self.wavlm_model.eval()
        with torch.no_grad():
            outputs = self.wavlm_model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        all_states = outputs.hidden_states[1:]
        group_outputs, group_probs, final_weights = self.adaptive_pool(all_states)

        bsz = group_outputs[0].size(0)
        T = group_outputs[0].size(1)
        cls = self.cls_token.expand(bsz, -1, -1)
        g0 = torch.cat([cls, group_outputs[0]], dim=1)
        g1 = torch.cat([cls, group_outputs[1]], dim=1)
        g0 = self.positional_encoding(g0)
        g1 = self.positional_encoding(g1)

        key_padding_mask = self._downsampled_mask(attention_mask, T)
        if key_padding_mask is not None:
            cls_pad = torch.zeros((bsz, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad.bool(), key_padding_mask.bool()], dim=1)
        else:
            key_padding_mask = None

        final_output = self.co_attention_module(g0, g1, key_padding_mask=key_padding_mask)
        seq_no_cls = final_output[:, 1:, :]
        ctc_logits = self.ctc_classifier(seq_no_cls)

        cls_repr = self.pre_class_norm(final_output[:, 0, :])
        logits = self.classifier(cls_repr).squeeze(-1)
        return logits, ctc_logits, group_probs, final_weights

# ───────────── CTC 目标生成函数 ─────────────
def generate_hubert_policy_targets_online(
    raw_waves,
    labels,
    k,
    device,
    ctc_input_lengths,
    hubert_model,
):
    """Build pseudo tokens from valid HuBERT frames only."""
    waves_np = [w.cpu().numpy() for w in raw_waves]
    hubert_inputs = hubert_feature_extractor(
        waves_np,
        sampling_rate=16000,
        return_tensors='pt',
        padding=True,
        return_attention_mask=True,
    )
    hubert_input_values = hubert_inputs.input_values
    hubert_attention_mask = lengths_to_attention_mask(
        torch.tensor([wave.numel() for wave in raw_waves]),
        hubert_input_values.size(1),
    )
    if device.type == 'cuda' and not hubert_input_values.is_pinned():
        hubert_input_values = hubert_input_values.pin_memory()
        hubert_attention_mask = hubert_attention_mask.pin_memory()
    hubert_input_values = hubert_input_values.to(device, non_blocking=True)
    hubert_attention_mask = hubert_attention_mask.to(
        device, non_blocking=True
    )
    with torch.no_grad():
        outputs = hubert_model(
            hubert_input_values,
            attention_mask=hubert_attention_mask,
            output_hidden_states=True,
        )
        features_b = outputs.hidden_states[12]
        hubert_lengths = feature_output_lengths(
            hubert_attention_mask.sum(dim=-1),
            hubert_model.config.conv_kernel,
            hubert_model.config.conv_stride,
        ).clamp_max(features_b.size(1))
        ctc_input_lengths = ctc_input_lengths.to(
            device=device, dtype=torch.long
        )
        if not torch.equal(hubert_lengths, ctc_input_lengths):
            raise RuntimeError(
                "HuBERT/WavLM valid feature lengths differ: "
                f"{hubert_lengths.tolist()} vs {ctc_input_lengths.tolist()}"
            )
        cluster_ids_b = masked_kmeans_batched(
            features_b, hubert_lengths, k
        )
        flat_targets, target_lengths = build_ctc_targets(
            cluster_ids_b,
            hubert_lengths,
            labels,
            k,
            CTC_TARGET_MODE,
        )
    if torch.any(target_lengths > ctc_input_lengths):
        raise RuntimeError(
            "CTC target length exceeds its valid WavLM input length"
        )
    diagnostics = {
        'input_length_sum': float(ctc_input_lengths.sum().item()),
        'target_length_sum': float(target_lengths.sum().item()),
        'examples': int(target_lengths.numel()),
        'valid_feature_fraction': float(
            ctc_input_lengths.sum().item()
            / max(1, ctc_input_lengths.numel() * features_b.size(1))
        ),
    }
    return (
        flat_targets,
        ctc_input_lengths,
        target_lengths,
        diagnostics,
    )


def generate_cached_global_targets(
    paths,
    crop_starts,
    labels,
    ctc_input_lengths,
    cache,
    k,
    target_mode,
    temporal_policy,
    device,
    time_steps,
    unit_stride=1,
    reverse=False,
    require_aligned=False,
):
    """Load stable fold-specific units for CTC or aligned framewise CE."""
    sequences = []
    lengths = ctc_input_lengths.to(device=device, dtype=torch.long)
    for path, crop_start, frame_length in zip(
        paths, crop_starts, lengths.tolist()
    ):
        sequence = cache.crop_sequence(
            path,
            crop_start_samples=int(crop_start),
            frame_length=int(frame_length),
            require_aligned=bool(require_aligned),
        ).to(device)
        if torch.any(sequence < 0) or torch.any(sequence >= int(k)):
            raise RuntimeError(f"Out-of-range global unit in {path}")
        sequences.append(sequence)
    max_length = max(int(sequence.numel()) for sequence in sequences)
    padded = torch.zeros(
        (len(sequences), max_length), dtype=torch.long, device=device
    )
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.numel()] = sequence
    flat_targets, target_lengths = build_ctc_targets(
        padded,
        lengths,
        labels,
        k,
        target_mode,
        unit_stride=int(unit_stride),
        reverse=bool(reverse),
    )
    frame_targets = None
    frame_lengths = torch.div(
        lengths + int(unit_stride) - 1,
        int(unit_stride),
        rounding_mode='floor',
    )
    if temporal_policy == 'global_units_frame_ce':
        if reverse:
            raise ValueError("Reverse targets are only defined for CTC")
        frame_time_steps = (
            int(time_steps) + int(unit_stride) - 1
        ) // int(unit_stride)
        frame_targets = torch.full(
            (len(sequences), frame_time_steps),
            -100,
            dtype=torch.long,
            device=device,
        )
        for index, sequence in enumerate(sequences):
            aligned = sequence[:: int(unit_stride)]
            if target_mode == 'label_shifted' and int(labels[index]) == 1:
                aligned = aligned + int(k)
            frame_targets[index, : aligned.numel()] = aligned
        # CE uses every valid frame; the collapsed length remains useful for
        # side-by-side temporal geometry diagnostics.
    diagnostics = {
        'input_length_sum': float(lengths.sum().item()),
        'target_length_sum': float(
            (
                frame_lengths
                if temporal_policy == 'global_units_frame_ce'
                else target_lengths
            ).sum().item()
        ),
        'examples': int(lengths.numel()),
        'valid_feature_fraction': float(
            lengths.sum().item()
            / max(1, lengths.numel() * int(time_steps))
        ),
    }
    return (
        flat_targets,
        lengths,
        target_lengths,
        frame_targets,
        frame_lengths,
        diagnostics,
    )

# ───────────── subject 级聚合（test 用） ─────────────
def aggregate_predictions_by_subject(pred_probs, subject_ids, labels):
    """Aggregate utterance probabilities under the configured policy."""
    scores, subject_labels, _subjects = aggregate_subject_predictions(
        pred_probs,
        subject_ids,
        labels,
        policy=AGGREGATION_MODE,
    )
    return np.asarray(scores), np.asarray(subject_labels)

# ───────────── Loss / Optim ─────────────
criterion = nn.BCEWithLogitsLoss()


def create_opt(model, lr, total_steps):
    if OPTIMIZER_POLICY == 'legacy_adamw':
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable, lr=lr, weight_decay=WEIGHT_DECAY
        )
        return optimizer, None, {
            'decay_parameters': sum(p.numel() for p in trainable),
            'no_decay_parameters': 0,
        }
    parameter_groups, group_counts = adamw_parameter_groups(
        model, WEIGHT_DECAY
    )
    optimizer = optim.AdamW(parameter_groups, lr=lr)
    warmup_steps = int(round(float(total_steps) * LR_WARMUP_RATIO))
    warmup_steps = min(max(1, warmup_steps), int(total_steps) - 1)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_multiplier(
            step,
            total_steps=int(total_steps),
            warmup_steps=warmup_steps,
            min_ratio=LR_MIN_RATIO,
        ),
    )
    return optimizer, scheduler, group_counts

AMP_ENABLED = device.type == 'cuda' and AMP_DTYPE_POLICY != 'fp32'
AMP_DTYPE = (
    torch.bfloat16 if AMP_DTYPE_POLICY == 'bf16' else torch.float16
)
def create_grad_scaler():
    return torch.amp.GradScaler(
        'cuda',
        enabled=AMP_ENABLED and AMP_DTYPE_POLICY == 'fp16',
    )
ctc_loss_fn = nn.CTCLoss(
    blank=2 * CTC_K, zero_infinity=False
)  # invalid alignments must fail visibly
neutral_ctc_loss_fn = nn.CTCLoss(
    blank=CTC_K, zero_infinity=False
)

# Loaded after DataLoader workers fork so workers never inherit a CUDA context.
hubert_model = None
global_unit_cache = (
    PackedUnitCache(GLOBAL_UNIT_CACHE, expected_k=CTC_K)
    if (
        CTC_ENABLED
        and TEMPORAL_TARGET_POLICY.startswith('global_units_')
        and (SPLIT_MODE != 'eval_only' or TEMPORAL_EVAL_DIAGNOSTICS)
    )
    else None
)
if global_unit_cache is not None:
    cache_metadata = global_unit_cache.metadata
    cache_ids = set(global_unit_cache.identifiers)
    cache_layout = (
        cache_metadata.get('codebook_fit_split'),
        cache_metadata.get('cached_roles'),
    )
    accepted_cache_layouts = {
        ('exact_fold_train_manifest_ids_only', ('train', 'dev')),
        ('exact_outer_train_dev_manifest_ids_only', ('train_dev',)),
    }
    if (
        int(cache_metadata.get('frame_stride_samples', -1)) != 320
        or (cache_layout[0], tuple(cache_layout[1] or ()))
        not in accepted_cache_layouts
        or cache_metadata.get('excluded_roles') != ['test']
        or (
            FOLD_INDEX in range(5)
            and int(cache_metadata.get('fold', -1)) != FOLD_INDEX
        )
    ):
        raise ValueError("Global-unit cache provenance/frame grid is invalid")
    if (
        CTC_LOSS_POLICY == 'normalized_fp32'
        and cache_metadata.get('codebook_sampling_policy')
        != 'exact_train_id_subject_equal_chunk_uniform_frames_v2'
    ):
        raise ValueError(
            "normalized_fp32 CTC requires a subject-balanced global codebook"
        )
    if (
        int(cache_metadata.get('cached_utterances', -1)) != len(cache_ids)
        or cache_metadata.get('cached_identifier_sha256')
        != identifier_set_sha256(cache_ids)
    ):
        raise ValueError("Global-unit cache identifier provenance is invalid")
    if TRAIN_WAVS:
        required_unit_ids = {Path(path).stem for path in TRAIN_WAVS}
        missing_unit_ids = required_unit_ids - cache_ids
        if missing_unit_ids:
            raise RuntimeError(
                "Global-unit cache lacks training utterances: "
                f"{sorted(missing_unit_ids)[:20]}"
            )
        if (
            cache_metadata.get('train_identifier_sha256')
            != identifier_set_sha256(required_unit_ids)
            or (
                TRAIN_MANIFEST
                and cache_metadata.get('manifest_sha256', {}).get('train')
                != _cache_file_sha256(Path(TRAIN_MANIFEST))
            )
        ):
            raise RuntimeError(
                "Global-unit cache does not match the active train manifest"
            )
    if VAL_WAVS and SPLIT_MODE != 'test_tune':
        validation_unit_ids = {Path(path).stem for path in VAL_WAVS}
        if (
            cache_metadata.get('dev_identifier_sha256')
            != identifier_set_sha256(validation_unit_ids)
            or (
                VAL_MANIFEST
                and cache_metadata.get('manifest_sha256', {}).get('dev')
                != _cache_file_sha256(Path(VAL_MANIFEST))
            )
        ):
            raise RuntimeError(
                "Global-unit cache does not match the active dev manifest"
            )
    if VAL_WAVS and SPLIT_MODE == 'test_tune':
        heldout_unit_ids = {Path(path).stem for path in VAL_WAVS}
        if heldout_unit_ids & cache_ids:
            raise RuntimeError(
                "Test-tuning heldout evaluation must be cache-disjoint"
            )
    if SPLIT_MODE == 'eval_only':
        evaluation_unit_ids = {Path(path).stem for path in TEST_WAVS}
        if TEMPORAL_EVAL_DIAGNOSTICS:
            if (
                cache_metadata.get('dev_identifier_sha256')
                != identifier_set_sha256(evaluation_unit_ids)
                or not evaluation_unit_ids.issubset(cache_ids)
            ):
                raise RuntimeError(
                    "Temporal diagnostics require the cache's exact dev split"
                )
        elif evaluation_unit_ids & cache_ids:
            raise RuntimeError(
                "Non-diagnostic held-out evaluation must be cache-disjoint"
            )

# ───────────── Multi-run training loop ─────────────
from sklearn.metrics import classification_report as _clsrep

LOG_INTERVAL = int(os.environ.get('LOG_INTERVAL', '25'))
METRIC_NAMES = [
    'f1_macro',
    'f1_pos',
    'f1_neg',
    'precision_macro',
    'recall_macro',
    'sens',
    'spec',
    'auc',
]

with open(os.path.join(LOG_DIR, 'run_config.json'), 'w', encoding='utf-8') as config_file:
    json.dump(
        {
            'data_root': _DATA_ROOT,
            'seeds': SEEDS,
            'epochs': NUM_EPOCHS,
            'schedule_epochs': SCHEDULE_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'ctc_enabled': CTC_ENABLED,
            'ctc_weight': CTC_WEIGHT,
            'ctc_mode': CTC_MODE,
            'ctc_target_mode': CTC_TARGET_MODE,
            'ctc_target_ratio': CTC_TARGET_RATIO,
            'ctc_grad_target_ratio': CTC_GRAD_TARGET_RATIO,
            'ctc_grad_update_interval': CTC_GRAD_UPDATE_INTERVAL,
            'ctc_grad_ema_decay': CTC_GRAD_EMA_DECAY,
            'ctc_warmup_ratio': CTC_WARMUP_RATIO,
            'ctc_loss_policy': CTC_LOSS_POLICY,
            'temporal_head_policy': TEMPORAL_HEAD_POLICY,
            'global_unit_stride': GLOBAL_UNIT_STRIDE,
            'global_unit_reverse': GLOBAL_UNIT_REVERSE,
            'ctc_warmup_epochs': CTC_WARMUP_EPOCHS,
            'ctc_clusters': CTC_K,
            'wavlm_mask_policy': WAVLM_MASK_POLICY,
            'wavlm_preprocess_policy': WAVLM_PREPROCESS_POLICY,
            'wavlm_batch_padding_policy': WAVLM_BATCH_PADDING_POLICY,
            'train_crop_policy': TRAIN_CROP_POLICY,
            'training_rng_policy': 'reseed_after_model_initialization',
            'wavlm_input_policy': (
                f'{WAVLM_PREPROCESS_POLICY}_'
                f'{WAVLM_BATCH_PADDING_POLICY}_{WAVLM_MASK_POLICY}_mask'
            ),
            'routing_init_policy': ROUTING_INIT_POLICY,
            'sampling_policy': SAMPLING_POLICY,
            'temporal_target_policy': TEMPORAL_TARGET_POLICY,
            'global_unit_cache': GLOBAL_UNIT_CACHE,
            'global_unit_cache_sha256': (
                global_unit_cache.sha256 if global_unit_cache else ''
            ),
            'ctc_input_policy': (
                'true_length_hubert_masked_kmeans'
                if TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
                else 'true_length_cached_global_units'
            ),
            'gradient_diagnostic_interval': GRAD_DIAGNOSTIC_INTERVAL,
            'gradient_clip_norm': GRAD_CLIP_NORM,
            'gradient_clip_policy': GRAD_CLIP_POLICY,
            'crop_alignment_samples': CROP_ALIGNMENT_SAMPLES,
            'silence_threshold_dbfs': SILENCE_THRESHOLD_DBFS,
            'test_policy': TEST_POLICY,
            'eval_checkpoint': EVAL_CHECKPOINT,
            'protocol': TRAINING_PROTOCOL,
            'aggregation': AGGREGATION_MODE,
            'split_mode': SPLIT_MODE,
            'fold_index': FOLD_INDEX,
            'train_manifest': TRAIN_MANIFEST,
            'val_manifest': VAL_MANIFEST,
            'num_workers': NUM_WORKERS,
            'prefetch_factor': PREFETCH_FACTOR,
            'amp_enabled': AMP_ENABLED,
            'amp_dtype': AMP_DTYPE_POLICY,
            'optimizer_policy': OPTIMIZER_POLICY,
            'head_arch_policy': HEAD_ARCH_POLICY,
            'head_init_policy': HEAD_INIT_POLICY,
            'lr_warmup_ratio': LR_WARMUP_RATIO,
            'lr_min_ratio': LR_MIN_RATIO,
            'ema_decay': EMA_DECAY,
            'save_training_state': SAVE_TRAINING_STATE,
            'resume_checkpoint': RESUME_CHECKPOINT,
            'eval_crop_policy': EVAL_CROP_POLICY,
            'eval_window_stride_seconds': EVAL_WINDOW_STRIDE_SECONDS,
            'temporal_eval_diagnostics': TEMPORAL_EVAL_DIAGNOSTICS,
            'sampler': (
                'not_applicable' if EVAL_CHECKPOINT else SAMPLER_NAME
            ),
            'run_tag': RUN_TAG,
            'checkpoint_selection': (
                'pre_frozen_checkpoint'
                if EVAL_CHECKPOINT
                else (
                    'global_test_epoch_search'
                    if SPLIT_MODE == 'test_tune'
                    else (
                        (
                            'train_only_fixed_epoch'
                            if SPLIT_MODE == 'train_only'
                            else 'fixed_final_epoch'
                        )
                        if SPLIT_MODE in {'cv', 'train_only'}
                        else 'dev_subject_metrics'
                    )
                )
            ),
            'checkpoint_tiebreakers': [
                'dev_f1_macro',
                'dev_auc',
                'dev_f1_pos',
                'dev_sensitivity',
                'earlier_epoch',
            ] if SPLIT_MODE == 'fixed' and not EVAL_CHECKPOINT else [],
            'wavlm_model': 'microsoft/wavlm-large',
            'wavlm_model_revision': '',
            'hubert_model': (
                'facebook/hubert-large-ll60k'
                if (
                    CTC_ENABLED
                    and TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
                )
                else None
            ),
        },
        config_file,
        indent=2,
        sort_keys=True,
    )

all_epoch_results = []
all_run_results = []


def _record_wavlm_revision(revision):
    if not revision or revision == 'unresolved':
        raise RuntimeError("WavLM artifact revision could not be resolved")
    path = os.path.join(LOG_DIR, 'run_config.json')
    with open(path, encoding='utf-8') as config_file:
        payload = json.load(config_file)
    previous = payload.get('wavlm_model_revision', '')
    if previous not in {'', revision}:
        raise RuntimeError("WavLM revision changed within one experiment")
    payload['wavlm_model_revision'] = revision
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as config_file:
        json.dump(payload, config_file, indent=2, sort_keys=True)
        config_file.write('\n')
    os.replace(temporary, path)


def _loader_kwargs():
    kwargs = {
        'num_workers': NUM_WORKERS,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        kwargs['prefetch_factor'] = PREFETCH_FACTOR
    return kwargs


def _metric_fields(prefix, result):
    return {
        f'{prefix}_{name}': float(result.get(name, 0.0))
        for name in METRIC_NAMES
    }


def _selection_score(result, epoch):
    # Dev-only selection. Test metrics are deliberately absent.
    return (
        float(result['f1_macro']),
        float(result.get('auc', 0.0)),
        float(result['f1_pos']),
        float(result['sens']),
        -int(epoch),
    )


def _checkpoint_metrics(result):
    saved = {}
    for key, value in result.items():
        if key.startswith('_'):
            continue
        if key == 'cm':
            saved[key] = np.asarray(value).tolist()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            saved[key] = float(value)
    return saved


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(
    model,
    path,
    seed,
    epoch,
    evaluation_result,
    selection_policy,
    initialization_hash,
    shared_initialization_hash,
    trainable_state_override=None,
):
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if trainable_state_override is None:
        model_state = model.state_dict()
        trainable_state = {
            name: model_state[name].detach().cpu()
            for name in sorted(trainable_names)
        }
    else:
        if set(trainable_state_override) != trainable_names:
            raise RuntimeError(
                "Checkpoint state override does not match trainable parameters"
            )
        trainable_state = {
            name: trainable_state_override[name].detach().cpu()
            for name in sorted(trainable_names)
        }
    payload = {
        'seed': int(seed),
        'fold': int(FOLD_INDEX),
        'epoch': int(epoch),
        'selection_policy': selection_policy,
        'trainable_initialization_sha256': initialization_hash,
        'shared_initialization_sha256': shared_initialization_hash,
        'weight_policy': 'ema' if trainable_state_override is not None else 'online',
        'evaluation_metrics': _checkpoint_metrics(evaluation_result),
        'trainable_state_dict': trainable_state,
        'config': {
            'protocol': TRAINING_PROTOCOL,
            'split_mode': SPLIT_MODE,
            'aggregation': AGGREGATION_MODE,
            'learning_rate': LR,
            'batch_size': BATCH_SIZE,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'ctc_enabled': CTC_ENABLED,
            'ctc_weight': CTC_WEIGHT,
            'ctc_mode': CTC_MODE,
            'ctc_target_mode': CTC_TARGET_MODE,
            'ctc_target_ratio': CTC_TARGET_RATIO,
            'ctc_grad_target_ratio': CTC_GRAD_TARGET_RATIO,
            'ctc_grad_update_interval': CTC_GRAD_UPDATE_INTERVAL,
            'ctc_grad_ema_decay': CTC_GRAD_EMA_DECAY,
            'ctc_warmup_ratio': CTC_WARMUP_RATIO,
            'ctc_loss_policy': CTC_LOSS_POLICY,
            'temporal_head_policy': TEMPORAL_HEAD_POLICY,
            'global_unit_stride': GLOBAL_UNIT_STRIDE,
            'global_unit_reverse': GLOBAL_UNIT_REVERSE,
            'ctc_warmup_epochs': CTC_WARMUP_EPOCHS,
            'ctc_clusters': CTC_K,
            'wavlm_mask_policy': WAVLM_MASK_POLICY,
            'wavlm_preprocess_policy': WAVLM_PREPROCESS_POLICY,
            'wavlm_batch_padding_policy': WAVLM_BATCH_PADDING_POLICY,
            'train_crop_policy': TRAIN_CROP_POLICY,
            'training_rng_policy': 'reseed_after_model_initialization',
            'wavlm_input_policy': (
                f'{WAVLM_PREPROCESS_POLICY}_'
                f'{WAVLM_BATCH_PADDING_POLICY}_{WAVLM_MASK_POLICY}_mask'
            ),
            'routing_init_policy': ROUTING_INIT_POLICY,
            'sampling_policy': SAMPLING_POLICY,
            'temporal_target_policy': TEMPORAL_TARGET_POLICY,
            'global_unit_cache': GLOBAL_UNIT_CACHE,
            'global_unit_cache_sha256': (
                global_unit_cache.sha256 if global_unit_cache else ''
            ),
            'ctc_input_policy': (
                'true_length_hubert_masked_kmeans'
                if TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
                else 'true_length_cached_global_units'
            ),
            'gradient_diagnostic_interval': GRAD_DIAGNOSTIC_INTERVAL,
            'gradient_clip_norm': GRAD_CLIP_NORM,
            'gradient_clip_policy': GRAD_CLIP_POLICY,
            'crop_alignment_samples': CROP_ALIGNMENT_SAMPLES,
            'amp_dtype': AMP_DTYPE_POLICY,
            'optimizer_policy': OPTIMIZER_POLICY,
            'head_arch_policy': HEAD_ARCH_POLICY,
            'head_init_policy': HEAD_INIT_POLICY,
            'lr_warmup_ratio': LR_WARMUP_RATIO,
            'lr_min_ratio': LR_MIN_RATIO,
            'ema_decay': EMA_DECAY,
            'sampler': SAMPLER_NAME,
            'test_policy': 'none',
            'num_workers': NUM_WORKERS,
            'prefetch_factor': PREFETCH_FACTOR,
            'epochs': NUM_EPOCHS,
            'schedule_epochs': SCHEDULE_EPOCHS,
            'data_root': _DATA_ROOT,
            'train_manifest': TRAIN_MANIFEST,
            'train_manifest_sha256': (
                _file_sha256(TRAIN_MANIFEST) if TRAIN_MANIFEST else ''
            ),
            'val_manifest': VAL_MANIFEST,
            'val_manifest_sha256': (
                _file_sha256(VAL_MANIFEST) if VAL_MANIFEST else ''
            ),
            'wavlm_model': 'microsoft/wavlm-large',
            'wavlm_model_revision': WAVLM_MODEL_REVISION,
            'hubert_model': (
                'facebook/hubert-large-ll60k'
                if TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
                else None
            ),
        },
    }
    temporary_path = path + '.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    metadata = {
        key: value
        for key, value in payload.items()
        if key != 'trainable_state_dict'
    }
    metadata['checkpoint_path'] = os.path.realpath(path)
    metadata['checkpoint_sha256'] = _file_sha256(path)
    metadata_path = path + '.metadata.json'
    temporary_metadata_path = metadata_path + '.tmp'
    with open(
        temporary_metadata_path, 'w', encoding='utf-8'
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write('\n')
    os.replace(temporary_metadata_path, metadata_path)


def _load_trainable_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    incompatible = model.load_state_dict(payload['trainable_state_dict'], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_backbone = [
        key for key in incompatible.missing_keys if not key.startswith('wavlm_model.')
    ]
    if unexpected or missing_non_backbone:
        raise RuntimeError(
            f"Checkpoint mismatch: unexpected={unexpected}, "
            f"missing_non_backbone={missing_non_backbone}"
        )
    return payload


def _resume_config():
    return {
        'schema_version': 1,
        'seed': int(SEEDS[0]) if len(SEEDS) == 1 else None,
        'fold': FOLD_INDEX,
        'split_mode': SPLIT_MODE,
        'epochs': NUM_EPOCHS,
        'schedule_epochs': SCHEDULE_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'ctc_enabled': CTC_ENABLED,
        'ctc_weight': CTC_WEIGHT,
        'ctc_mode': CTC_MODE,
        'ctc_target_mode': CTC_TARGET_MODE,
        'ctc_target_ratio': CTC_TARGET_RATIO,
        'ctc_grad_target_ratio': CTC_GRAD_TARGET_RATIO,
        'ctc_grad_update_interval': CTC_GRAD_UPDATE_INTERVAL,
        'ctc_grad_ema_decay': CTC_GRAD_EMA_DECAY,
        'ctc_warmup_ratio': CTC_WARMUP_RATIO,
        'ctc_loss_policy': CTC_LOSS_POLICY,
        'temporal_head_policy': TEMPORAL_HEAD_POLICY,
        'global_unit_stride': GLOBAL_UNIT_STRIDE,
        'global_unit_reverse': GLOBAL_UNIT_REVERSE,
        'ctc_warmup_epochs': CTC_WARMUP_EPOCHS,
        'ctc_clusters': CTC_K,
        'wavlm_mask_policy': WAVLM_MASK_POLICY,
        'wavlm_preprocess_policy': WAVLM_PREPROCESS_POLICY,
        'wavlm_batch_padding_policy': WAVLM_BATCH_PADDING_POLICY,
        'train_crop_policy': TRAIN_CROP_POLICY,
        'training_rng_policy': 'reseed_after_model_initialization',
        'routing_init_policy': ROUTING_INIT_POLICY,
        'sampling_policy': SAMPLING_POLICY,
        'aggregation': AGGREGATION_MODE,
        'temporal_target_policy': TEMPORAL_TARGET_POLICY,
        'global_unit_cache_sha256': (
            global_unit_cache.sha256 if global_unit_cache else ''
        ),
        'gradient_clip_norm': GRAD_CLIP_NORM,
        'gradient_clip_policy': GRAD_CLIP_POLICY,
        'crop_alignment_samples': CROP_ALIGNMENT_SAMPLES,
        'amp_dtype': AMP_DTYPE_POLICY,
        'optimizer_policy': OPTIMIZER_POLICY,
        'head_arch_policy': HEAD_ARCH_POLICY,
        'head_init_policy': HEAD_INIT_POLICY,
        'lr_warmup_ratio': LR_WARMUP_RATIO,
        'lr_min_ratio': LR_MIN_RATIO,
        'ema_decay': EMA_DECAY,
        'data_root': os.path.realpath(_DATA_ROOT),
        'train_manifest_sha256': (
            _file_sha256(TRAIN_MANIFEST) if TRAIN_MANIFEST else ''
        ),
        'val_manifest_sha256': (
            _file_sha256(VAL_MANIFEST) if VAL_MANIFEST else ''
        ),
    }


def _save_training_state(
    *,
    model,
    optimizer,
    scheduler,
    ema,
    ctc_controller,
    path,
    run_id,
    seed,
    completed_epoch,
    selected_result,
    selected_epoch,
    initialization_hash,
    shared_initialization_hash,
    epoch_rows,
    evaluation_events,
):
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model_state = model.state_dict()
    payload = {
        'schema_version': 1,
        'run_id': run_id,
        'seed': int(seed),
        'fold': FOLD_INDEX,
        'completed_epoch': int(completed_epoch),
        'selected_epoch': int(selected_epoch),
        'selected_result': (
            _checkpoint_metrics(selected_result)
            if selected_result is not None
            else None
        ),
        'trainable_initialization_sha256': initialization_hash,
        'shared_initialization_sha256': shared_initialization_hash,
        'trainable_state_dict': {
            name: model_state[name].detach().cpu()
            for name in sorted(trainable_names)
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (
            scheduler.state_dict() if scheduler is not None else None
        ),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'ctc_controller_state_dict': (
            ctc_controller.state_dict()
            if ctc_controller is not None
            else None
        ),
        'rng_state': capture_rng_state(
            include_cuda=device.type == 'cuda'
        ),
        'epoch_rows': list(epoch_rows),
        'evaluation_events': list(evaluation_events),
        'resume_config': _resume_config(),
    }
    atomic_torch_save(payload, path)


def _restore_training_state(
    *,
    model,
    optimizer,
    scheduler,
    ema,
    ctc_controller,
    path,
    run_id,
    seed,
    initialization_hash,
    shared_initialization_hash,
):
    payload = _load_trainable_checkpoint(model, path)
    expected = _resume_config()
    if payload.get('schema_version') != 1:
        raise RuntimeError("Unsupported training-state schema")
    if (
        payload.get('run_id') != run_id
        or int(payload.get('seed', -1)) != int(seed)
        or int(payload.get('fold', -999)) != FOLD_INDEX
        or payload.get('resume_config') != expected
        or payload.get('trainable_initialization_sha256')
        != initialization_hash
        or payload.get('shared_initialization_sha256')
        != shared_initialization_hash
    ):
        raise RuntimeError("Training-state provenance/configuration mismatch")
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler_state = payload.get('scheduler_state_dict')
    if (scheduler is None) != (scheduler_state is None):
        raise RuntimeError("Training-state scheduler mismatch")
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler.load_state_dict(payload.get('scaler_state_dict', {}))
    ema_state = payload.get('ema_state_dict')
    if (ema is None) != (ema_state is None):
        raise RuntimeError("Training-state EMA mismatch")
    if ema is not None:
        ema.load_state_dict(ema_state, model)
    controller_state = payload.get('ctc_controller_state_dict')
    if (ctc_controller is None) != (controller_state is None):
        raise RuntimeError("Training-state CTC controller mismatch")
    if ctc_controller is not None:
        ctc_controller.load_state_dict(controller_state)
    restore_rng_state(
        payload['rng_state'], include_cuda=device.type == 'cuda'
    )
    return payload


for seed_idx, SEED in enumerate(SEEDS):
    run_id = (
        f'fold{FOLD_INDEX}_seed{SEED}'
        if SPLIT_MODE in {'cv', 'eval_only', 'test_tune'}
        else f'seed{SEED}'
    )
    seed_log_file = os.path.join(LOG_DIR, f"{LOG_BASE_NAME}_{run_id}.log")
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logging.root.removeHandler(handler)
            handler.close()
    seed_file_handler = logging.FileHandler(seed_log_file, mode='w')
    seed_file_handler.setLevel(logging.INFO)
    seed_file_handler.setFormatter(
        logging.Formatter(
            f'%(asctime)s - [{run_id}] - %(levelname)s - %(message)s'
        )
    )
    logging.root.addHandler(seed_file_handler)

    logging.info("\n%s", '=' * 80)
    logging.info(
        "Starting %s (%d/%d): mode=%s protocol=%s device=%s",
        run_id,
        seed_idx + 1,
        len(SEEDS),
        SPLIT_MODE,
        TRAINING_PROTOCOL,
        device,
    )
    logging.info(
        "batch=%d lr=%g wd=%g dropout=%g ctc=%s test_policy=%s "
        "workers=%d prefetch=%d amp=%s aggregation=%s wavlm_mask=%s "
        "wavlm_preprocess=%s wavlm_padding=%s train_crop=%s "
        "routing_init=%s sampling=%s temporal_target=%s amp_dtype=%s",
        BATCH_SIZE,
        LR,
        WEIGHT_DECAY,
        DROPOUT,
        CTC_ENABLED,
        TEST_POLICY,
        NUM_WORKERS,
        PREFETCH_FACTOR,
        AMP_ENABLED,
        AGGREGATION_MODE,
        WAVLM_MASK_POLICY,
        WAVLM_PREPROCESS_POLICY,
        WAVLM_BATCH_PADDING_POLICY,
        TRAIN_CROP_POLICY,
        ROUTING_INIT_POLICY,
        SAMPLING_POLICY,
        TEMPORAL_TARGET_POLICY,
        AMP_DTYPE_POLICY,
    )
    logging.info("%s\n", '=' * 80)

    # Seed CPU state first. CUDA seeding is deliberately deferred until after
    # worker processes start, avoiding CUDA's unsafe fork-after-init path.
    set_seed(SEED, seed_cuda=False)
    scaler = create_grad_scaler()
    label_encoder = LabelEncoder().fit([0, 1])
    train_ds = val_ds = test_ds = None
    if EVAL_CHECKPOINT:
        test_ds = AudioDataset(
            TEST_WAVS, feature_extractor, label_encoder, training=False
        )
        logging.info(
            "Test samples=%d classes=%s",
            len(test_ds),
            np.bincount(test_ds.labels, minlength=2).tolist(),
        )
    else:
        train_ds = AudioDataset(
            TRAIN_WAVS, feature_extractor, label_encoder, training=True
        )
        if SPLIT_MODE == 'train_only':
            logging.info(
                "Train-only samples=%d classes=%s; no evaluation split",
                len(train_ds),
                np.bincount(train_ds.labels, minlength=2).tolist(),
            )
        else:
            val_ds = AudioDataset(
                VAL_WAVS, feature_extractor, label_encoder, training=False
            )
            logging.info(
                "Train samples=%d classes=%s; val samples=%d classes=%s",
                len(train_ds),
                np.bincount(train_ds.labels, minlength=2).tolist(),
                len(val_ds),
                np.bincount(val_ds.labels, minlength=2).tolist(),
            )

    loader_kwargs = _loader_kwargs()
    tr_loader = val_loader = test_loader = None
    if EVAL_CHECKPOINT:
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=(
                collate_with_waves
                if TEMPORAL_EVAL_DIAGNOSTICS
                else collate
            ),
            **loader_kwargs,
        )
    else:
        if SAMPLING_POLICY == 'subject_class_balanced':
            sample_weights = subject_balanced_sample_weights(
                train_ds.labels, train_ds.subjects
            )
        else:
            class_counts = np.bincount(train_ds.labels, minlength=2)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            sample_weights = torch.tensor(
                [class_weights[label] for label in train_ds.labels],
                dtype=torch.double,
            )
        if TRAIN_CROP_POLICY == 'epoch_keyed':
            sampler = EpochSeededWeightedSampler(
                sample_weights,
                num_samples=len(sample_weights),
                seed=SEED,
            )
        else:
            loader_generator = torch.Generator().manual_seed(SEED)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=loader_generator,
            )
        tr_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            collate_fn=collate_with_waves,
            **loader_kwargs,
        )
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate,
                **loader_kwargs,
            )

    if NUM_WORKERS > 0:
        # DataLoader workers start lazily. Bootstrap them before any CUDA API
        # initializes a context in the parent process.
        if tr_loader is not None:
            iter(tr_loader)
        if val_loader is not None:
            iter(val_loader)
        if test_loader is not None:
            iter(test_loader)
        logging.info("Bootstrapped persistent DataLoader workers before CUDA init")

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
        logging.info(
            "GPU=%s capability=%s torch=%s cuda=%s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            torch.__version__,
            torch.version.cuda,
        )

    if (
        CTC_ENABLED
        and TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc'
        and not EVAL_CHECKPOINT
        and hubert_model is None
    ):
        hubert_model = HubertModel.from_pretrained(
            "facebook/hubert-large-ll60k"
        ).to(device)
        hubert_model.eval()

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    WAVLM_MODEL_REVISION = str(
        getattr(wavlm.config, "_commit_hash", None) or "unresolved"
    )
    _record_wavlm_revision(WAVLM_MODEL_REVISION)
    # Pretrained-model construction advances RNG state. Reset immediately
    # before trainable-head construction so paired CTC-on/off runs start from
    # byte-identical heads for the same seed.
    set_seed(SEED, seed_cuda=device.type == 'cuda')
    model = WavLMClassificationModel(
        wavlm,
        num_labels=1,
        num_groups=2,
        dropout=DROPOUT,
        ctc_weight=CTC_WEIGHT,
        k=CTC_K,
        routing_init_policy=ROUTING_INIT_POLICY,
        head_arch_policy=HEAD_ARCH_POLICY,
        head_init_policy=HEAD_INIT_POLICY,
        temporal_head_policy=TEMPORAL_HEAD_POLICY,
        initialization_seed=SEED,
    ).to(device)
    model.wavlm_model.eval()
    for parameter in model.wavlm_model.parameters():
        parameter.requires_grad = False
    initialization_hash = trainable_state_hash(model)
    shared_initialization_hash = named_parameter_hash(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not name.startswith('ctc_classifier.')
        )
    )
    paired_core_initialization_hash = named_parameter_hash(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                name == 'cls_token'
                or name.startswith('adaptive_pool.')
                or name.startswith('pre_class_norm.')
                or name.startswith('classifier.')
                or name.startswith('co_attention_module.attention.')
                or name.startswith('co_attention_module.norm1.')
                or name.startswith('co_attention_module.norm2.')
            )
        )
    )
    initialization_path = os.path.join(
        LOG_DIR, f'trainable_init_seed{SEED}.sha256'
    )
    with open(initialization_path, 'w', encoding='utf-8') as init_file:
        init_file.write(initialization_hash + '\n')
    logging.info(
        "Trainable initialization seed=%d sha256=%s shared_sha256=%s "
        "paired_core_sha256=%s",
        SEED,
        initialization_hash,
        shared_initialization_hash,
        paired_core_initialization_hash,
    )
    # Temporal-head shape (K+1) consumes a K-dependent amount of RNG during
    # construction. Reset after all parameter initialization so paired K=10
    # and K=50 runs receive identical subsequent dropout streams.
    set_seed(SEED, seed_cuda=device.type == 'cuda')
    logging.info(
        "Training RNG reset after model initialization for exact pairing"
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logging.info(
        "Trainable params=%s / total=%s (WavLM frozen+eval)",
        f"{n_trainable:,}",
        f"{n_total:,}",
    )
    if EVAL_CHECKPOINT:
        optimizer = scheduler = ema = None
        optimizer_group_counts = {
            'decay_parameters': 0,
            'no_decay_parameters': 0,
        }
    else:
        optimizer, scheduler, optimizer_group_counts = create_opt(
            model, LR, len(tr_loader) * SCHEDULE_EPOCHS
        )
        ema = TrainableEMA(model, EMA_DECAY) if EMA_DECAY > 0 else None
        logging.info(
            "optimizer=%s decay_params=%d no_decay_params=%d "
            "warmup_ratio=%.3f min_lr_ratio=%.3f ema_decay=%.5f",
            OPTIMIZER_POLICY,
            optimizer_group_counts['decay_parameters'],
            optimizer_group_counts['no_decay_parameters'],
            LR_WARMUP_RATIO,
            LR_MIN_RATIO,
            EMA_DECAY,
        )
    ctc_controller = (
        SharedGradientRatioController(
            target_ratio=CTC_GRAD_TARGET_RATIO,
            max_weight=CTC_WEIGHT,
            loss_ratio_cap=CTC_TARGET_RATIO,
            warmup_steps=int(
                round(len(tr_loader) * SCHEDULE_EPOCHS * CTC_WARMUP_RATIO)
            ),
            update_interval=CTC_GRAD_UPDATE_INTERVAL,
            ema_decay=CTC_GRAD_EMA_DECAY,
        )
        if CTC_ENABLED
        and CTC_MODE == 'shared_grad_norm'
        and not EVAL_CHECKPOINT
        else None
    )
    initial_routing_probabilities = (
        torch.softmax(
            model.adaptive_pool.group_assignment.detach(), dim=-1
        ).cpu()
    )
    shared_temporal_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name == 'cls_token'
            or name.startswith('adaptive_pool.')
            or name.startswith('positional_encoding.')
            or name.startswith('co_attention_module.')
        )
    ]
    primary_parameter_group = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith('ctc_classifier.')
    ]
    temporal_head_parameter_group = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith('ctc_classifier.')
    ]

    def _eval_loader(loader):
        probabilities, labels, subjects, paths_seen = [], [], [], []
        temporal_edit_distance = 0
        temporal_target_tokens = 0
        temporal_predicted_tokens = 0
        temporal_exact_sequences = 0
        temporal_sequences = 0
        parameter_context = (
            ema.average_parameters(model)
            if ema is not None
            else contextlib.nullcontext()
        )
        with parameter_context:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    if TEMPORAL_EVAL_DIAGNOSTICS:
                        (
                            xb,
                            attention_mask,
                            yb,
                            _phq_scores,
                            _phq_severities,
                            paths,
                            subjects_in_batch,
                            raw_waves,
                            crop_starts,
                        ) = batch
                    else:
                        (
                            xb,
                            attention_mask,
                            yb,
                            _phq_scores,
                            _phq_severities,
                            paths,
                            subjects_in_batch,
                        ) = batch
                    xb = xb.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(
                        device, non_blocking=True
                    )
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=AMP_ENABLED,
                        dtype=AMP_DTYPE,
                    ):
                        cls_logits, ctc_logits, _, _ = model(
                            xb, attention_mask=attention_mask
                        )
                    if TEMPORAL_EVAL_DIAGNOSTICS:
                        true_sample_lengths = torch.tensor(
                            [wave.numel() for wave in raw_waves],
                            dtype=torch.long,
                            device=device,
                        )
                        input_lengths = (
                            model.ctc_input_lengths_from_samples(
                                true_sample_lengths
                            ).clamp_max(ctc_logits.size(1))
                        )
                        (
                            flat_targets,
                            input_lengths,
                            target_lengths,
                            _frame_targets,
                            _frame_lengths,
                            _target_diagnostics,
                        ) = generate_cached_global_targets(
                            paths,
                            crop_starts,
                            yb.long(),
                            input_lengths,
                            global_unit_cache,
                            model.k,
                            CTC_TARGET_MODE,
                            TEMPORAL_TARGET_POLICY,
                            device,
                            ctc_logits.size(1),
                            unit_stride=GLOBAL_UNIT_STRIDE,
                            reverse=False,
                            require_aligned=True,
                        )
                        temporal_logits, blank_index = temporal_logit_view(
                            ctc_logits,
                            k=model.k,
                            target_mode=CTC_TARGET_MODE,
                            framewise=False,
                        )
                        counts = ctc_greedy_edit_counts(
                            temporal_logits,
                            flat_targets,
                            input_lengths,
                            target_lengths,
                            blank=int(blank_index),
                        )
                        temporal_edit_distance += counts["edit_distance"]
                        temporal_target_tokens += counts["target_tokens"]
                        temporal_predicted_tokens += counts[
                            "predicted_tokens"
                        ]
                        temporal_exact_sequences += counts[
                            "exact_sequences"
                        ]
                        temporal_sequences += counts["sequences"]
                    batch_probabilities = (
                        torch.sigmoid(cls_logits)
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    probabilities.extend(batch_probabilities)
                    labels.extend(yb.numpy())
                    subjects.extend(subjects_in_batch)
                    paths_seen.extend(paths)
        physical_window_count = len(paths_seen)
        if EVAL_CROP_POLICY in {'multi3', 'sliding_all'}:
            (
                paths_seen,
                subjects,
                labels,
                probabilities,
            ) = collapse_sliding_window_predictions(
                paths_seen,
                subjects,
                labels,
                probabilities,
            )
        aggregated_probabilities, aggregated_labels = aggregate_predictions_by_subject(
            probabilities,
            subjects,
            labels,
        )
        aggregated_labels = aggregated_labels.astype(int)
        aggregated_predictions = (
            np.asarray(aggregated_probabilities) >= 0.5
        ).astype(int)
        result = metrics(
            aggregated_labels,
            aggregated_predictions,
            y_probs=aggregated_probabilities,
        )
        report = _clsrep(
            aggregated_labels,
            aggregated_predictions,
            target_names=['neg(0)', 'pos(1)'],
            digits=4,
            zero_division=0,
        )
        result['_utt'] = [
            (os.path.basename(str(path))[:-4], int(label), float(probability))
            for path, label, probability in zip(paths_seen, labels, probabilities)
        ]
        result['_physical_window_count'] = physical_window_count
        if TEMPORAL_EVAL_DIAGNOSTICS:
            result['_ctc_unit_error_rate'] = (
                temporal_edit_distance / max(1, temporal_target_tokens)
            )
            result['_ctc_exact_sequence_fraction'] = (
                temporal_exact_sequences / max(1, temporal_sequences)
            )
            result['_ctc_edit_distance'] = temporal_edit_distance
            result['_ctc_target_tokens'] = temporal_target_tokens
            result['_ctc_predicted_tokens'] = temporal_predicted_tokens
            result['_ctc_sequences'] = temporal_sequences
        return result, report

    evaluation_events = []
    evaluation_events_path = os.path.join(
        LOG_DIR, f'evaluation_events_seed{SEED}.json'
    )
    if SPLIT_MODE == 'train_only':
        with open(
            evaluation_events_path, 'w', encoding='utf-8'
        ) as events_file:
            json.dump([], events_file)
            events_file.write('\n')

    def _recorded_eval_loader(loader, split, purpose, epoch):
        result, report = _eval_loader(loader)
        event = {
            'invocation': len(evaluation_events) + 1,
            'split': split,
            'purpose': purpose,
            'epoch': int(epoch),
            'utterances': len(result.get('_utt', [])),
            'physical_windows': int(
                result.get(
                    '_physical_window_count',
                    len(result.get('_utt', [])),
                )
            ),
            'subjects': len(
                {
                    str(utt_id).split('_')[0]
                    for utt_id, _label, _prob in result.get('_utt', [])
                }
            ),
        }
        if TEMPORAL_EVAL_DIAGNOSTICS:
            event['temporal_diagnostics'] = {
                'unit_error_rate': float(result['_ctc_unit_error_rate']),
                'exact_sequence_fraction': float(
                    result['_ctc_exact_sequence_fraction']
                ),
                'edit_distance': int(result['_ctc_edit_distance']),
                'target_tokens': int(result['_ctc_target_tokens']),
                'predicted_tokens': int(result['_ctc_predicted_tokens']),
                'sequences': int(result['_ctc_sequences']),
            }
        evaluation_events.append(event)
        with open(
            evaluation_events_path, 'w', encoding='utf-8'
        ) as events_file:
            json.dump(evaluation_events, events_file, indent=2)
            events_file.write('\n')
        return result, report

    if EVAL_CHECKPOINT:
        checkpoint_payload = _load_trainable_checkpoint(model, EVAL_CHECKPOINT)
        if int(checkpoint_payload['seed']) != int(SEED):
            raise RuntimeError(
                f"Checkpoint seed {checkpoint_payload['seed']} != requested {SEED}"
            )
        checkpoint_config = checkpoint_payload.get('config', {})
        checkpoint_split_mode = checkpoint_config.get('split_mode')
        if (
            SPLIT_MODE == 'eval_only'
            and checkpoint_split_mode not in {'inner', 'test_tune'}
        ):
            raise RuntimeError(
                "Fold evaluation requires an inner or fixed-epoch test_tune "
                "checkpoint"
            )
        expected_config = {
            'protocol': TRAINING_PROTOCOL,
            'split_mode': (
                checkpoint_split_mode if SPLIT_MODE == 'eval_only' else 'fixed'
            ),
            'aggregation': AGGREGATION_MODE,
            'learning_rate': LR,
            'batch_size': BATCH_SIZE,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'ctc_enabled': CTC_ENABLED,
            'ctc_weight': CTC_WEIGHT,
            'ctc_mode': CTC_MODE,
            'ctc_target_mode': CTC_TARGET_MODE,
            'ctc_target_ratio': CTC_TARGET_RATIO,
            'ctc_grad_target_ratio': CTC_GRAD_TARGET_RATIO,
            'ctc_grad_update_interval': CTC_GRAD_UPDATE_INTERVAL,
            'ctc_grad_ema_decay': CTC_GRAD_EMA_DECAY,
            'ctc_warmup_ratio': CTC_WARMUP_RATIO,
            'ctc_loss_policy': CTC_LOSS_POLICY,
            'temporal_head_policy': TEMPORAL_HEAD_POLICY,
            'global_unit_stride': GLOBAL_UNIT_STRIDE,
            'global_unit_reverse': GLOBAL_UNIT_REVERSE,
            'ctc_warmup_epochs': CTC_WARMUP_EPOCHS,
            'ctc_clusters': CTC_K,
            'wavlm_mask_policy': WAVLM_MASK_POLICY,
            'wavlm_preprocess_policy': WAVLM_PREPROCESS_POLICY,
            'wavlm_batch_padding_policy': WAVLM_BATCH_PADDING_POLICY,
            'train_crop_policy': TRAIN_CROP_POLICY,
            'routing_init_policy': ROUTING_INIT_POLICY,
            'sampling_policy': SAMPLING_POLICY,
            'temporal_target_policy': TEMPORAL_TARGET_POLICY,
            'global_unit_cache_sha256': (
                global_unit_cache.sha256 if global_unit_cache else ''
            ),
            'amp_dtype': AMP_DTYPE_POLICY,
            'gradient_clip_policy': GRAD_CLIP_POLICY,
            'crop_alignment_samples': CROP_ALIGNMENT_SAMPLES,
            'optimizer_policy': OPTIMIZER_POLICY,
            'head_arch_policy': HEAD_ARCH_POLICY,
            'head_init_policy': HEAD_INIT_POLICY,
            'lr_warmup_ratio': LR_WARMUP_RATIO,
            'lr_min_ratio': LR_MIN_RATIO,
            'ema_decay': EMA_DECAY,
            'sampler': SAMPLER_NAME,
            'test_policy': 'none',
            'epochs': NUM_EPOCHS,
            'schedule_epochs': SCHEDULE_EPOCHS,
            'wavlm_model': 'microsoft/wavlm-large',
            'wavlm_model_revision': WAVLM_MODEL_REVISION,
        }
        if SPLIT_MODE == 'eval_only' and not TEMPORAL_EVAL_DIAGNOSTICS:
            expected_config.pop('global_unit_cache_sha256')
        legacy_config_defaults = {
            'wavlm_mask_policy': 'legacy_full',
            'wavlm_preprocess_policy': 'legacy_prepad',
            'wavlm_batch_padding_policy': 'fixed_10s',
            'train_crop_policy': 'worker_random',
            'routing_init_policy': 'legacy',
            'sampling_policy': 'utterance_class_balanced',
            'temporal_target_policy': 'local_kmeans_ctc',
            'global_unit_cache_sha256': '',
            'amp_dtype': 'fp16',
            'optimizer_policy': 'legacy_adamw',
            'head_arch_policy': 'legacy_17m',
            'head_init_policy': 'legacy_stream',
            'ctc_grad_target_ratio': 0.1,
            'ctc_grad_update_interval': 10,
            'ctc_grad_ema_decay': 0.9,
            'ctc_warmup_ratio': 0.1,
            'ctc_loss_policy': 'legacy_mean',
            'temporal_head_policy': 'legacy_2k1',
            'global_unit_stride': 1,
            'global_unit_reverse': False,
            'gradient_clip_policy': 'global',
            'crop_alignment_samples': 1,
            'lr_warmup_ratio': 0.1,
            'lr_min_ratio': 0.1,
            'ema_decay': 0.0,
            'schedule_epochs': NUM_EPOCHS,
            'wavlm_model_revision': WAVLM_MODEL_REVISION,
        }
        for key, expected in expected_config.items():
            actual = checkpoint_config.get(
                key, legacy_config_defaults.get(key)
            )
            if actual != expected:
                raise RuntimeError(
                    f"Checkpoint config mismatch for {key}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        checkpoint_data_root = os.path.realpath(
            checkpoint_config.get('data_root', '')
        )
        if (
            SPLIT_MODE != 'eval_only'
            and checkpoint_data_root != os.path.realpath(_DATA_ROOT)
        ):
            raise RuntimeError(
                "Checkpoint data_root does not match the requested test variant"
            )
        if (
            SPLIT_MODE == 'eval_only'
            and checkpoint_config.get('val_manifest_sha256')
            != _file_sha256(VAL_MANIFEST)
        ):
            raise RuntimeError(
                "Checkpoint held-out manifest does not match evaluation data"
            )
        allowed_selection_policies = (
            {'dev_subject_metrics', 'global_test_epoch_search'}
            if SPLIT_MODE == 'eval_only'
            else {'dev_subject_metrics'}
        )
        if (
            checkpoint_payload.get('selection_policy')
            not in allowed_selection_policies
        ):
            raise RuntimeError("Checkpoint selection policy is not evaluable")
        if (
            SPLIT_MODE == 'eval_only'
            and int(checkpoint_payload.get('fold', -1)) != FOLD_INDEX
        ):
            raise RuntimeError("Fold-evaluation checkpoint fold mismatch")
        evaluation_label = (
            'tuning_holdout' if SPLIT_MODE == 'eval_only' else 'test'
        )
        evaluation_purpose = (
            'frozen_checkpoint_evaluation'
            if SPLIT_MODE == 'eval_only'
            else 'final_checkpoint_evaluation'
        )
        test_result, test_report = _recorded_eval_loader(
            test_loader,
            split=evaluation_label,
            purpose=evaluation_purpose,
            epoch=int(checkpoint_payload['epoch']),
        )
        test_report_path = os.path.join(
            REPORT_DIR, f'{run_id}_{evaluation_label}.txt'
        )
        with open(test_report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(test_report)
            report_file.write(
                f"\nROC-AUC: {test_result.get('auc', 0.0):.6f}\n"
            )
            report_file.write(
                f"Checkpoint: {EVAL_CHECKPOINT}\n"
                f"Checkpoint epoch: {int(checkpoint_payload['epoch'])}\n"
            )
            if TEMPORAL_EVAL_DIAGNOSTICS:
                report_file.write(
                    "CTC unit error rate: "
                    f"{test_result['_ctc_unit_error_rate']:.6f}\n"
                    "CTC exact sequence fraction: "
                    f"{test_result['_ctc_exact_sequence_fraction']:.6f}\n"
                )
        prediction_path = os.path.join(
            LOG_DIR, f'{evaluation_label}_pred_{run_id}.csv'
        )
        pd.DataFrame(
            test_result.get('_utt', []),
            columns=['utt_id', 'label', 'prob'],
        ).to_csv(prediction_path, index=False)
        selection_metrics = checkpoint_payload.get('evaluation_metrics', {})
        run_row = {
            'run_id': run_id,
            'seed': SEED,
            'fold': int(checkpoint_payload.get('fold', -1)),
            'split_mode': SPLIT_MODE,
            'protocol': TRAINING_PROTOCOL,
            'selection_epoch': int(checkpoint_payload['epoch']),
            'selection_policy': checkpoint_payload.get(
                'selection_policy', 'dev_subject_metrics'
            ),
            'trainable_initialization_sha256': checkpoint_payload.get(
                'trainable_initialization_sha256', ''
            ),
            'shared_initialization_sha256': checkpoint_payload.get(
                'shared_initialization_sha256',
                checkpoint_payload.get('trainable_initialization_sha256', ''),
            ),
            'checkpoint_path': EVAL_CHECKPOINT,
            'evaluation_split': evaluation_label,
            'evaluation_invocations': len(evaluation_events),
            'protected_evaluation_invocations': 1,
            **_metric_fields('dev', selection_metrics),
            **_metric_fields('eval', test_result),
        }
        if TEMPORAL_EVAL_DIAGNOSTICS:
            run_row.update(
                {
                    'eval_ctc_unit_error_rate': test_result[
                        '_ctc_unit_error_rate'
                    ],
                    'eval_ctc_exact_sequence_fraction': test_result[
                        '_ctc_exact_sequence_fraction'
                    ],
                    'eval_ctc_edit_distance': test_result[
                        '_ctc_edit_distance'
                    ],
                    'eval_ctc_target_tokens': test_result[
                        '_ctc_target_tokens'
                    ],
                    'eval_ctc_predicted_tokens': test_result[
                        '_ctc_predicted_tokens'
                    ],
                    'eval_ctc_sequences': test_result['_ctc_sequences'],
                }
            )
        all_run_results.append(run_row)
        pd.DataFrame(all_run_results).to_csv(PER_RUN_CSV, index=False)
        log_metrics(f"{evaluation_label.upper()}-{run_id}", test_result)
        del model, wavlm, test_loader
        torch.cuda.empty_cache()
        continue

    selection_policy = (
        'train_only_fixed_epoch'
        if SPLIT_MODE == 'train_only'
        else (
            'global_test_epoch_search'
            if SPLIT_MODE == 'test_tune'
            else (
                'fixed_final_epoch'
                if SPLIT_MODE == 'cv'
                else 'dev_subject_metrics'
            )
        )
    )
    checkpoint_name = (
        f'{run_id}_epoch{NUM_EPOCHS:02d}.pt'
        if SPLIT_MODE in {'cv', 'train_only', 'test_tune'}
        else f'{run_id}_best_dev.pt'
    )
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    selected_result = None
    selected_epoch = -1
    start_epoch = 1
    training_state_path = os.path.join(
        CHECKPOINT_DIR, f'{run_id}_last_training_state.pt'
    )
    if RESUME_CHECKPOINT:
        resume_payload = _restore_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            ctc_controller=ctc_controller,
            path=RESUME_CHECKPOINT,
            run_id=run_id,
            seed=SEED,
            initialization_hash=initialization_hash,
            shared_initialization_hash=shared_initialization_hash,
        )
        completed_epoch = int(resume_payload['completed_epoch'])
        if completed_epoch >= NUM_EPOCHS:
            raise RuntimeError("Training state already completed all epochs")
        selected_result = resume_payload.get('selected_result')
        selected_epoch = int(resume_payload.get('selected_epoch', -1))
        all_epoch_results.extend(resume_payload.get('epoch_rows', []))
        evaluation_events[:] = resume_payload.get(
            'evaluation_events', []
        )
        with open(
            evaluation_events_path, 'w', encoding='utf-8'
        ) as events_file:
            json.dump(evaluation_events, events_file, indent=2)
            events_file.write('\n')
        if selected_epoch >= 0 and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Best checkpoint is missing beside resumed output: "
                f"{checkpoint_path}"
            )
        start_epoch = completed_epoch + 1
        logging.info(
            "Resumed exact epoch-boundary state from %s; next_epoch=%d",
            RESUME_CHECKPOINT,
            start_epoch,
        )

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        logging.info("\n%s", '=' * 60)
        logging.info("%s - Epoch %d/%d", run_id, epoch, NUM_EPOCHS)
        logging.info("%s", '=' * 60)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        if TRAIN_CROP_POLICY == 'epoch_keyed':
            sampler.set_epoch(epoch)

        model.train()
        epoch_train_started = time.perf_counter()
        data_wait_seconds = 0.0
        batch_wait_started = epoch_train_started
        train_examples = 0
        cls_loss_accum = 0.0
        ctc_loss_accum = 0.0
        weighted_ctc_loss_accum = 0.0
        ctc_ratio_accum = 0.0
        max_ctc_ratio = 0.0
        effective_weight_accum = 0.0
        ctc_batches = 0
        ctc_positive_batches = 0
        ctc_zero_loss_batches = 0
        ctc_input_length_sum = 0.0
        ctc_target_length_sum = 0.0
        ctc_examples = 0
        valid_feature_fraction_sum = 0.0
        wavlm_attention_fraction_sum = 0.0
        speech_fraction_sum = 0.0
        waveform_rms_sum = 0.0
        waveform_quality_examples = 0
        subject_exposure = {}
        subject_exposure_labels = {}
        primary_gradient_norm_sum = 0.0
        auxiliary_gradient_norm_sum = 0.0
        weighted_auxiliary_gradient_norm_sum = 0.0
        weighted_auxiliary_to_primary_ratio_sum = 0.0
        gradient_cosine_sum = 0.0
        gradient_conflicts = 0
        gradient_diagnostic_batches = 0
        gradient_norm_sum = 0.0
        primary_group_gradient_norm_sum = 0.0
        temporal_head_gradient_norm_sum = 0.0
        finite_gradient_norm_batches = 0
        gradient_clipped_batches = 0
        primary_group_clipped_batches = 0
        temporal_head_clipped_batches = 0
        nonfinite_gradient_batches = 0
        amp_overflow_batches = 0
        crop_schedule_digest = hashlib.sha256()
        n_batches = 0
        for (
            xb,
            attention_mask,
            yb,
            _phq_scores,
            _phq_severities,
            paths,
            subject_ids,
            raw_waves,
            crop_starts,
        ) in tr_loader:
            data_wait_seconds += time.perf_counter() - batch_wait_started
            train_examples += int(yb.numel())
            xb = xb.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            yb = yb.float().to(device, non_blocking=True)
            wavlm_attention_fraction_sum += float(
                attention_mask.float().mean().item()
            )
            batch_labels = yb.detach().to("cpu").to(torch.int64).tolist()
            for subject_id, label in zip(subject_ids, batch_labels):
                key = str(subject_id)
                subject_exposure[key] = subject_exposure.get(key, 0) + 1
                prior_label = subject_exposure_labels.setdefault(
                    key, int(label)
                )
                if prior_label != int(label):
                    raise RuntimeError(
                        f"Subject {key} has inconsistent sampled labels"
                    )
            for path, crop_start in zip(paths, crop_starts):
                crop_schedule_digest.update(
                    f"{path}:{int(crop_start)}\n".encode("utf-8")
                )
            for raw_wave in raw_waves:
                quality = waveform_quality_metrics(
                    raw_wave,
                    speech_threshold_dbfs=SILENCE_THRESHOLD_DBFS,
                )
                waveform_rms_sum += quality['rms']
                speech_fraction_sum += quality['speech_frame_fraction']
                waveform_quality_examples += 1
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=AMP_ENABLED,
                dtype=AMP_DTYPE,
            ):
                cls_logits, ctc_logits, _group_probs, _ = model(
                    xb, attention_mask=attention_mask
                )
                cls_loss = criterion(cls_logits, yb)
                _, time_steps, _ = ctc_logits.shape
                current_ctc_weight = 0.0
                ctc_loss = torch.zeros((), device=xb.device)
                ctc_diagnostics = None
                gradient_balance_diagnostic = None
                if CTC_ENABLED:
                    true_sample_lengths = torch.tensor(
                        [wave.numel() for wave in raw_waves],
                        dtype=torch.long,
                        device=device,
                    )
                    ctc_input_lengths = model.ctc_input_lengths_from_samples(
                        true_sample_lengths
                    ).clamp_max(time_steps)
                    if model.ctc_weight > 0:
                        frame_targets = None
                        frame_lengths = None
                        if TEMPORAL_TARGET_POLICY == 'local_kmeans_ctc':
                            (
                                flat_targets,
                                input_lengths,
                                target_lengths,
                                ctc_diagnostics,
                            ) = generate_hubert_policy_targets_online(
                                raw_waves,
                                yb.long(),
                                model.k,
                                device,
                                ctc_input_lengths,
                                hubert_model,
                            )
                        else:
                            (
                                flat_targets,
                                input_lengths,
                                target_lengths,
                                frame_targets,
                                frame_lengths,
                                ctc_diagnostics,
                            ) = generate_cached_global_targets(
                                paths,
                                crop_starts,
                                yb.long(),
                                ctc_input_lengths,
                                global_unit_cache,
                                model.k,
                                CTC_TARGET_MODE,
                                TEMPORAL_TARGET_POLICY,
                                device,
                                time_steps,
                                unit_stride=GLOBAL_UNIT_STRIDE,
                                reverse=GLOBAL_UNIT_REVERSE,
                                require_aligned=(
                                    CROP_ALIGNMENT_SAMPLES % 320 == 0
                                ),
                            )
                        if TEMPORAL_TARGET_POLICY == 'global_units_frame_ce':
                            frame_logits, _blank = temporal_logit_view(
                                ctc_logits,
                                k=model.k,
                                target_mode=CTC_TARGET_MODE,
                                framewise=True,
                            )
                            frame_logits = frame_logits[
                                :, ::GLOBAL_UNIT_STRIDE, :
                            ]
                            if CTC_LOSS_POLICY == 'normalized_fp32':
                                ctc_loss, _per_example_temporal_loss = (
                                    normalized_frame_ce_loss(
                                        frame_logits,
                                        frame_targets,
                                        frame_lengths,
                                    )
                                )
                            else:
                                ctc_loss = F.cross_entropy(
                                    frame_logits.reshape(
                                        -1, frame_logits.size(-1)
                                    ),
                                    frame_targets.reshape(-1),
                                    ignore_index=-100,
                                )
                        elif flat_targets.numel() > 0:
                            temporal_logits, blank_index = temporal_logit_view(
                                ctc_logits,
                                k=model.k,
                                target_mode=CTC_TARGET_MODE,
                                framewise=False,
                            )
                            temporal_loss_fn = (
                                neutral_ctc_loss_fn
                                if blank_index == model.k
                                else ctc_loss_fn
                            )
                            if CTC_LOSS_POLICY == 'normalized_fp32':
                                ctc_loss, _per_example_temporal_loss = (
                                    normalized_ctc_loss(
                                        temporal_logits,
                                        flat_targets,
                                        input_lengths,
                                        target_lengths,
                                        blank=int(blank_index),
                                    )
                                )
                            else:
                                ctc_loss = temporal_loss_fn(
                                    F.log_softmax(
                                        temporal_logits.transpose(0, 1), dim=-1
                                    ),
                                    flat_targets,
                                    input_lengths,
                                    target_lengths,
                                )
                        if not torch.isfinite(ctc_loss):
                            raise RuntimeError(
                                "Non-finite temporal loss; "
                                f"policy={TEMPORAL_TARGET_POLICY} "
                                f"input_lengths={input_lengths.tolist()} "
                                f"target_lengths={target_lengths.tolist()}"
                            )
                        if float(ctc_loss.detach().item()) > 0:
                            if ctc_controller is not None:
                                (
                                    current_ctc_weight,
                                    gradient_balance_diagnostic,
                                ) = ctc_controller.weight(
                                    primary_loss=cls_loss,
                                    auxiliary_loss=ctc_loss,
                                    shared_parameters=(
                                        shared_temporal_parameters
                                    ),
                                    step_index=(
                                        (epoch - 1) * len(tr_loader)
                                        + n_batches
                                    ),
                                )
                            else:
                                warmup_factor = (
                                    1.0
                                    if CTC_WARMUP_EPOCHS == 0
                                    else min(
                                        1.0,
                                        epoch / CTC_WARMUP_EPOCHS,
                                    )
                                )
                                current_ctc_weight = effective_ctc_weight(
                                    cls_loss=cls_loss,
                                    ctc_loss=ctc_loss,
                                    mode=CTC_MODE,
                                    max_weight=model.ctc_weight,
                                    warmup_factor=warmup_factor,
                                    target_ratio=CTC_TARGET_RATIO,
                                )
                weighted_ctc_loss = current_ctc_weight * ctc_loss
                loss = cls_loss + weighted_ctc_loss

            balance_diagnostic_due = (
                gradient_balance_diagnostic is not None
                and bool(
                    gradient_balance_diagnostic[
                        'gradient_balance_updated'
                    ]
                )
            )
            diagnostic_due = balance_diagnostic_due or (
                CTC_ENABLED
                and GRAD_DIAGNOSTIC_INTERVAL > 0
                and float(ctc_loss.detach().item()) > 0
                and (
                    n_batches == 0
                    or (n_batches + 1) % GRAD_DIAGNOSTIC_INTERVAL == 0
                )
            )
            if diagnostic_due:
                gradient_diagnostic = (
                    gradient_balance_diagnostic
                    if balance_diagnostic_due
                    else loss_gradient_diagnostics(
                        cls_loss,
                        ctc_loss,
                        shared_temporal_parameters,
                    )
                )
                primary_gradient_norm_sum += gradient_diagnostic[
                    'primary_gradient_norm'
                ]
                auxiliary_gradient_norm_sum += gradient_diagnostic[
                    'auxiliary_gradient_norm'
                ]
                weighted_auxiliary_gradient_norm_sum += (
                    gradient_diagnostic['auxiliary_gradient_norm']
                    * current_ctc_weight
                )
                weighted_auxiliary_to_primary_ratio_sum += (
                    gradient_diagnostic['auxiliary_gradient_norm']
                    * current_ctc_weight
                    / max(
                        gradient_diagnostic['primary_gradient_norm'],
                        1e-12,
                    )
                )
                gradient_cosine_sum += gradient_diagnostic['gradient_cosine']
                gradient_conflicts += int(
                    gradient_diagnostic['gradient_cosine'] < 0
                )
                gradient_diagnostic_batches += 1
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                if GRAD_CLIP_POLICY == 'task_grouped':
                    primary_group_norm = torch.nn.utils.clip_grad_norm_(
                        primary_parameter_group, GRAD_CLIP_NORM
                    )
                    temporal_head_norm = torch.nn.utils.clip_grad_norm_(
                        temporal_head_parameter_group, GRAD_CLIP_NORM
                    )
                    primary_group_norm_value = float(
                        primary_group_norm.detach().item()
                    )
                    temporal_head_norm_value = float(
                        temporal_head_norm.detach().item()
                    )
                    gradient_norm_value = float(
                        np.hypot(
                            primary_group_norm_value,
                            temporal_head_norm_value,
                        )
                    )
                else:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        GRAD_CLIP_NORM,
                    )
                    gradient_norm_value = float(
                        gradient_norm.detach().item()
                    )
                    primary_group_norm_value = gradient_norm_value
                    temporal_head_norm_value = 0.0
                if np.isfinite(gradient_norm_value):
                    gradient_norm_sum += gradient_norm_value
                    primary_group_gradient_norm_sum += (
                        primary_group_norm_value
                    )
                    temporal_head_gradient_norm_sum += (
                        temporal_head_norm_value
                    )
                    finite_gradient_norm_batches += 1
                    primary_was_clipped = (
                        primary_group_norm_value > GRAD_CLIP_NORM
                    )
                    temporal_was_clipped = (
                        temporal_head_norm_value > GRAD_CLIP_NORM
                    )
                    primary_group_clipped_batches += int(
                        primary_was_clipped
                    )
                    temporal_head_clipped_batches += int(
                        temporal_was_clipped
                    )
                    gradient_clipped_batches += int(
                        primary_was_clipped or temporal_was_clipped
                    )
                else:
                    nonfinite_gradient_batches += 1
            amp_scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_skipped = scaler.get_scale() < amp_scale_before
            if optimizer_step_skipped:
                amp_overflow_batches += 1
            else:
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            cls_loss_accum += float(cls_loss.detach().item())
            ctc_loss_accum += float(ctc_loss.detach().item())
            weighted_value = float(weighted_ctc_loss.detach().item())
            weighted_ctc_loss_accum += weighted_value
            batch_ctc_ratio = weighted_value / max(
                float(cls_loss.detach().item()), 1e-12
            )
            ctc_ratio_accum += batch_ctc_ratio
            max_ctc_ratio = max(max_ctc_ratio, batch_ctc_ratio)
            effective_weight_accum += current_ctc_weight
            if ctc_diagnostics is not None:
                ctc_batches += 1
                if float(ctc_loss.detach().item()) > 0:
                    ctc_positive_batches += 1
                else:
                    ctc_zero_loss_batches += 1
                examples = int(ctc_diagnostics['examples'])
                ctc_examples += examples
                ctc_input_length_sum += ctc_diagnostics['input_length_sum']
                ctc_target_length_sum += ctc_diagnostics[
                    'target_length_sum'
                ]
                valid_feature_fraction_sum += (
                    ctc_diagnostics['valid_feature_fraction'] * examples
                )
            n_batches += 1
            if LOG_INTERVAL > 0 and n_batches % LOG_INTERVAL == 0:
                logging.info(
                    "%s epoch=%d batch=%d/%d cls_loss=%.4f "
                    "ctc_loss=%.4f weighted_ctc=%.4f ratio=%.4f",
                    run_id,
                    epoch,
                    n_batches,
                    len(tr_loader),
                    cls_loss_accum / n_batches,
                    ctc_loss_accum / n_batches,
                    weighted_ctc_loss_accum / n_batches,
                    ctc_ratio_accum / n_batches,
                )
            batch_wait_started = time.perf_counter()

        train_wall_seconds = time.perf_counter() - epoch_train_started
        train_cls_loss_mean = cls_loss_accum / max(1, n_batches)
        train_ctc_loss_mean = ctc_loss_accum / max(1, n_batches)
        train_weighted_ctc_loss_mean = (
            weighted_ctc_loss_accum / max(1, n_batches)
        )
        train_ctc_ratio_mean = ctc_ratio_accum / max(1, n_batches)
        effective_ctc_weight_mean = (
            effective_weight_accum / max(1, n_batches)
        )
        routing_probabilities = torch.softmax(
            model.adaptive_pool.group_assignment.detach(), dim=-1
        ).cpu()
        routing_mean_abs_change = float(
            (routing_probabilities - initial_routing_probabilities)
            .abs()
            .mean()
            .item()
        )
        routing_entropy = float(
            (
                -routing_probabilities
                * routing_probabilities.clamp_min(1e-12).log()
            )
            .sum(dim=-1)
            .div(np.log(2.0))
            .mean()
            .item()
        )
        exposure_values = np.asarray(
            list(subject_exposure.values()), dtype=float
        )
        exposure_by_label = {
            label: np.asarray(
                [
                    subject_exposure[subject]
                    for subject, observed_label in subject_exposure_labels.items()
                    if observed_label == label
                ],
                dtype=float,
            )
            for label in (0, 1)
        }
        if CTC_ENABLED and (
            ctc_batches != n_batches
            or ctc_positive_batches != ctc_batches
            or ctc_zero_loss_batches != 0
            or train_weighted_ctc_loss_mean <= 0
        ):
            raise RuntimeError(
                "CTC-on epoch contained missing/zero auxiliary updates: "
                f"batches={n_batches} ctc_batches={ctc_batches} "
                f"positive={ctc_positive_batches} "
                f"zero={ctc_zero_loss_batches} "
                f"weighted_mean={train_weighted_ctc_loss_mean}"
            )
        logging.info(
            "[%s-Epoch%d] train_cls_loss=%.4f train_ctc_loss=%.4f "
            "weighted_ctc=%.4f ctc_to_bce=%.4f",
            run_id,
            epoch,
            train_cls_loss_mean,
            train_ctc_loss_mean,
            train_weighted_ctc_loss_mean,
            train_ctc_ratio_mean,
        )

        should_evaluate = (
            SPLIT_MODE in {'fixed', 'inner', 'test_tune'}
            or (SPLIT_MODE == 'cv' and epoch == NUM_EPOCHS)
        )
        val_result = val_report = None
        is_checkpoint_epoch = False
        if should_evaluate:
            split_label = (
                'dev'
                if SPLIT_MODE == 'fixed'
                else (
                    'inner_val'
                    if SPLIT_MODE == 'inner'
                    else (
                        'test_tuning'
                        if SPLIT_MODE == 'test_tune'
                        else 'held_out'
                    )
                )
            )
            val_result, val_report = _recorded_eval_loader(
                val_loader,
                split=split_label,
                purpose=(
                    'global_fixed_epoch_tuning'
                    if SPLIT_MODE == 'test_tune'
                    else 'epoch_evaluation'
                ),
                epoch=epoch,
            )
            val_report_path = os.path.join(
                REPORT_DIR, f'{run_id}_epoch{epoch:02d}_{split_label}.txt'
            )
            with open(val_report_path, 'w', encoding='utf-8') as report_file:
                report_file.write(val_report)
                report_file.write(
                    f"\nROC-AUC: {val_result.get('auc', 0.0):.6f}\n"
                )
            logging.info(
                "[%s-Epoch%d] === %s === macroF1=%.4f AUC=%.4f\n%s",
                run_id,
                epoch,
                split_label.upper(),
                val_result['f1_macro'],
                val_result.get('auc', 0.0),
                val_report,
            )
            if SPLIT_MODE == 'test_tune':
                epoch_prediction_path = os.path.join(
                    LOG_DIR,
                    f'{split_label}_pred_{run_id}_epoch{epoch:02d}.csv',
                )
                pd.DataFrame(
                    val_result.get('_utt', []),
                    columns=['utt_id', 'label', 'prob'],
                ).to_csv(epoch_prediction_path, index=False)
            is_checkpoint_epoch = (
                epoch == NUM_EPOCHS
                if SPLIT_MODE in {'cv', 'test_tune'}
                else (
                    selected_result is None
                    or _selection_score(val_result, epoch)
                    > _selection_score(selected_result, selected_epoch)
                )
            )
            if is_checkpoint_epoch:
                for prior_row in all_epoch_results:
                    if prior_row['run_id'] == run_id:
                        prior_row['is_checkpoint_epoch'] = False
                        prior_row['is_best_dev'] = False
                selected_result = copy.deepcopy(val_result)
                selected_epoch = epoch
                _save_checkpoint(
                    model,
                    checkpoint_path,
                    seed=SEED,
                    epoch=epoch,
                    evaluation_result=val_result,
                    selection_policy=selection_policy,
                    initialization_hash=initialization_hash,
                    shared_initialization_hash=shared_initialization_hash,
                    trainable_state_override=(
                        ema.shadow if ema is not None else None
                    ),
                )
                logging.info(
                    "[%s] saved %s checkpoint at epoch %d: %s",
                    run_id,
                    selection_policy,
                    epoch,
                    checkpoint_path,
                )
        elif SPLIT_MODE == 'train_only' and epoch == NUM_EPOCHS:
            is_checkpoint_epoch = True
            selected_result = {}
            selected_epoch = epoch
            _save_checkpoint(
                model,
                checkpoint_path,
                seed=SEED,
                epoch=epoch,
                evaluation_result={},
                selection_policy=selection_policy,
                initialization_hash=initialization_hash,
                shared_initialization_hash=shared_initialization_hash,
                trainable_state_override=(
                    ema.shadow if ema is not None else None
                ),
            )
            logging.info(
                "[%s] saved train-only checkpoint at epoch %d: %s",
                run_id,
                epoch,
                checkpoint_path,
            )

        epoch_row = {
            'run_id': run_id,
            'seed': SEED,
            'fold': FOLD_INDEX,
            'split_mode': SPLIT_MODE,
            'protocol': TRAINING_PROTOCOL,
            'epoch': epoch,
            'trainable_initialization_sha256': initialization_hash,
            'shared_initialization_sha256': shared_initialization_hash,
            'paired_core_initialization_sha256': (
                paired_core_initialization_hash
            ),
            'trainable_parameters': n_trainable,
            'head_arch_policy': HEAD_ARCH_POLICY,
            'head_init_policy': HEAD_INIT_POLICY,
            'selection_policy': selection_policy,
            'is_checkpoint_epoch': is_checkpoint_epoch,
            'is_best_dev': (
                is_checkpoint_epoch and SPLIT_MODE in {'fixed', 'inner'}
            ),
            'train_cls_loss': train_cls_loss_mean,
            'train_ctc_loss': train_ctc_loss_mean,
            'train_weighted_ctc_loss': train_weighted_ctc_loss_mean,
            'ctc_to_bce_ratio': train_ctc_ratio_mean,
            'max_ctc_to_bce_ratio': max_ctc_ratio,
            'ctc_weight': effective_ctc_weight_mean,
            'ctc_weight_max': CTC_WEIGHT,
            'ctc_mode': CTC_MODE,
            'ctc_target_mode': CTC_TARGET_MODE,
            'ctc_target_ratio': CTC_TARGET_RATIO,
            'ctc_grad_target_ratio': CTC_GRAD_TARGET_RATIO,
            'ctc_grad_update_interval': CTC_GRAD_UPDATE_INTERVAL,
            'ctc_grad_ema_decay': CTC_GRAD_EMA_DECAY,
            'ctc_warmup_ratio': CTC_WARMUP_RATIO,
            'ctc_loss_policy': CTC_LOSS_POLICY,
            'temporal_head_policy': TEMPORAL_HEAD_POLICY,
            'global_unit_stride': GLOBAL_UNIT_STRIDE,
            'global_unit_reverse': GLOBAL_UNIT_REVERSE,
            'ctc_batches': ctc_batches,
            'ctc_positive_batches': ctc_positive_batches,
            'ctc_zero_loss_batches': ctc_zero_loss_batches,
            'mean_ctc_input_length': (
                ctc_input_length_sum / max(1, ctc_examples)
            ),
            'mean_ctc_target_length': (
                ctc_target_length_sum / max(1, ctc_examples)
            ),
            'valid_feature_fraction': (
                valid_feature_fraction_sum / max(1, ctc_examples)
            ),
            'wavlm_attention_fraction': (
                wavlm_attention_fraction_sum / max(1, n_batches)
            ),
            'waveform_rms': (
                waveform_rms_sum / max(1, waveform_quality_examples)
            ),
            'speech_frame_fraction': (
                speech_fraction_sum / max(1, waveform_quality_examples)
            ),
            'subject_exposure_min': (
                float(exposure_values.min()) if len(exposure_values) else 0.0
            ),
            'subject_exposure_max': (
                float(exposure_values.max()) if len(exposure_values) else 0.0
            ),
            'subject_exposure_cv': (
                float(exposure_values.std(ddof=0) / exposure_values.mean())
                if len(exposure_values) and exposure_values.mean() > 0
                else 0.0
            ),
            'subject_exposure_cv_label0': (
                float(
                    exposure_by_label[0].std(ddof=0)
                    / exposure_by_label[0].mean()
                )
                if len(exposure_by_label[0])
                and exposure_by_label[0].mean() > 0
                else 0.0
            ),
            'subject_exposure_cv_label1': (
                float(
                    exposure_by_label[1].std(ddof=0)
                    / exposure_by_label[1].mean()
                )
                if len(exposure_by_label[1])
                and exposure_by_label[1].mean() > 0
                else 0.0
            ),
            'crop_schedule_sha256': crop_schedule_digest.hexdigest(),
            'routing_mean_abs_change': routing_mean_abs_change,
            'routing_entropy': routing_entropy,
            'primary_gradient_norm': (
                primary_gradient_norm_sum
                / max(1, gradient_diagnostic_batches)
            ),
            'auxiliary_gradient_norm': (
                auxiliary_gradient_norm_sum
                / max(1, gradient_diagnostic_batches)
            ),
            'weighted_auxiliary_gradient_norm': (
                weighted_auxiliary_gradient_norm_sum
                / max(1, gradient_diagnostic_batches)
            ),
            'weighted_auxiliary_to_primary_gradient_ratio': (
                weighted_auxiliary_to_primary_ratio_sum
                / max(1, gradient_diagnostic_batches)
            ),
            'gradient_cosine': (
                gradient_cosine_sum / max(1, gradient_diagnostic_batches)
            ),
            'gradient_conflict_fraction': (
                gradient_conflicts / max(1, gradient_diagnostic_batches)
            ),
            'gradient_diagnostic_batches': gradient_diagnostic_batches,
            'gradient_norm_before_clip': (
                gradient_norm_sum / max(1, finite_gradient_norm_batches)
            ),
            'primary_group_gradient_norm_before_clip': (
                primary_group_gradient_norm_sum
                / max(1, finite_gradient_norm_batches)
            ),
            'temporal_head_gradient_norm_before_clip': (
                temporal_head_gradient_norm_sum
                / max(1, finite_gradient_norm_batches)
            ),
            'gradient_clipped_fraction': (
                gradient_clipped_batches / max(1, n_batches)
            ),
            'primary_group_clipped_fraction': (
                primary_group_clipped_batches / max(1, n_batches)
            ),
            'temporal_head_clipped_fraction': (
                temporal_head_clipped_batches / max(1, n_batches)
            ),
            'grad_clip_policy': GRAD_CLIP_POLICY,
            'nonfinite_gradient_fraction': (
                nonfinite_gradient_batches / max(1, n_batches)
            ),
            'amp_overflow_fraction': (
                amp_overflow_batches / max(1, n_batches)
            ),
            'peak_gpu_memory_gb': (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if device.type == 'cuda'
                else 0.0
            ),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'ema_updates': int(ema.updates) if ema is not None else 0,
            'train_wall_seconds': train_wall_seconds,
            'data_wait_seconds': data_wait_seconds,
            'examples_per_second': (
                train_examples / max(train_wall_seconds, 1e-12)
            ),
        }
        if val_result is not None:
            epoch_row.update(_metric_fields('val', val_result))
        all_epoch_results.append(epoch_row)
        pd.DataFrame(all_epoch_results).to_csv(PER_EPOCH_CSV, index=False)
        if SAVE_TRAINING_STATE:
            _save_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                ctc_controller=ctc_controller,
                path=training_state_path,
                run_id=run_id,
                seed=SEED,
                completed_epoch=epoch,
                selected_result=selected_result,
                selected_epoch=selected_epoch,
                initialization_hash=initialization_hash,
                shared_initialization_hash=shared_initialization_hash,
                epoch_rows=[
                    row
                    for row in all_epoch_results
                    if row['run_id'] == run_id
                ],
                evaluation_events=evaluation_events,
            )
            logging.info(
                "[%s] saved resumable epoch state: %s",
                run_id,
                training_state_path,
            )

    # The selected checkpoint already contains EMA weights when enabled.
    # Disable the live final-epoch EMA before checkpoint reload verification.
    ema = None
    checkpoint_payload = _load_trainable_checkpoint(model, checkpoint_path)
    if int(checkpoint_payload['epoch']) != selected_epoch:
        raise RuntimeError("Saved checkpoint epoch does not match selection policy")
    if SPLIT_MODE == 'train_only':
        if evaluation_events:
            raise RuntimeError("train_only mode must not evaluate any split")
        run_row = {
            'run_id': run_id,
            'seed': SEED,
            'fold': FOLD_INDEX,
            'split_mode': SPLIT_MODE,
            'protocol': TRAINING_PROTOCOL,
            'selection_epoch': selected_epoch,
            'selection_policy': selection_policy,
            'trainable_initialization_sha256': initialization_hash,
            'shared_initialization_sha256': shared_initialization_hash,
            'checkpoint_path': checkpoint_path,
            'evaluation_split': 'none',
            'evaluation_invocations': 0,
            'protected_evaluation_invocations': 0,
        }
        all_run_results.append(run_row)
        pd.DataFrame(all_run_results).to_csv(PER_RUN_CSV, index=False)
        del model, wavlm, optimizer, tr_loader
        torch.cuda.empty_cache()
        continue
    if SPLIT_MODE == 'test_tune':
        if (
            len(evaluation_events) != NUM_EPOCHS
            or [event['epoch'] for event in evaluation_events]
            != list(range(1, NUM_EPOCHS + 1))
            or {event['split'] for event in evaluation_events}
            != {'test_tuning'}
            or {event['purpose'] for event in evaluation_events}
            != {'global_fixed_epoch_tuning'}
        ):
            raise RuntimeError("Incomplete test-tuning evaluation sequence")
        run_row = {
            'run_id': run_id,
            'seed': SEED,
            'fold': FOLD_INDEX,
            'split_mode': SPLIT_MODE,
            'protocol': TRAINING_PROTOCOL,
            'selection_epoch': -1,
            'selection_policy': selection_policy,
            'epoch_candidates': f'1-{NUM_EPOCHS}',
            'completed_epoch': NUM_EPOCHS,
            'trainable_initialization_sha256': initialization_hash,
            'shared_initialization_sha256': shared_initialization_hash,
            'checkpoint_path': checkpoint_path,
            'evaluation_split': 'test_tuning',
            'evaluation_invocations': len(evaluation_events),
            'protected_evaluation_invocations': 0,
            'heldout_used_for_epoch_selection': True,
        }
        all_run_results.append(run_row)
        pd.DataFrame(all_run_results).to_csv(PER_RUN_CSV, index=False)
        del model, wavlm, optimizer, tr_loader, val_loader
        torch.cuda.empty_cache()
        continue
    if SPLIT_MODE == 'cv':
        # The outer holdout is protected evaluation data. It is evaluated
        # exactly once, at the final retrained epoch.
        reloaded_val_result = selected_result
    else:
        reload_split = 'dev' if SPLIT_MODE == 'fixed' else 'inner_val'
        reloaded_val_result, _ = _recorded_eval_loader(
            val_loader,
            split=reload_split,
            purpose='checkpoint_reload_verification',
            epoch=selected_epoch,
        )
        for metric_name in METRIC_NAMES:
            before = float(selected_result.get(metric_name, 0.0))
            after = float(reloaded_val_result.get(metric_name, 0.0))
            if not np.isclose(after, before, atol=1e-5, rtol=0.0):
                logging.warning(
                    "Checkpoint reload changed dev %s by %.8f: %.8f -> %.8f",
                    metric_name,
                    abs(after - before),
                    before,
                    after,
                )

    evaluation_result = reloaded_val_result
    evaluation_split = (
        'held_out_fold'
        if SPLIT_MODE == 'cv'
        else ('inner_val' if SPLIT_MODE == 'inner' else 'dev')
    )
    prediction_path = os.path.join(
        LOG_DIR, f'{evaluation_split}_pred_{run_id}.csv'
    )
    pd.DataFrame(
        evaluation_result.get('_utt', []),
        columns=['utt_id', 'label', 'prob'],
    ).to_csv(prediction_path, index=False)

    log_metrics(
        f"{evaluation_split.upper()}-{run_id} "
        f"(selection_epoch={selected_epoch}, policy={selection_policy})",
        evaluation_result,
    )
    run_row = {
        'run_id': run_id,
        'seed': SEED,
        'fold': FOLD_INDEX,
        'split_mode': SPLIT_MODE,
        'protocol': TRAINING_PROTOCOL,
        'selection_epoch': selected_epoch,
        'selection_policy': selection_policy,
        'trainable_initialization_sha256': initialization_hash,
        'shared_initialization_sha256': shared_initialization_hash,
        'paired_core_initialization_sha256': (
            paired_core_initialization_hash
        ),
        'trainable_parameters': n_trainable,
        'head_arch_policy': HEAD_ARCH_POLICY,
        'head_init_policy': HEAD_INIT_POLICY,
        'checkpoint_path': checkpoint_path,
        'evaluation_split': evaluation_split,
        'evaluation_invocations': len(evaluation_events),
        'protected_evaluation_invocations': (
            1 if SPLIT_MODE == 'cv' else 0
        ),
        **_metric_fields('dev', reloaded_val_result),
        **_metric_fields('eval', evaluation_result),
    }
    all_run_results.append(run_row)
    pd.DataFrame(all_run_results).to_csv(PER_RUN_CSV, index=False)

    del model, wavlm, optimizer, tr_loader, val_loader
    torch.cuda.empty_cache()

if all_epoch_results:
    pd.DataFrame(all_epoch_results).to_csv(PER_EPOCH_CSV, index=False)
run_df = pd.DataFrame(all_run_results)
run_df.to_csv(PER_RUN_CSV, index=False)
logging.info("\nPer-epoch results: %s", PER_EPOCH_CSV)
logging.info("Per-run selected results: %s", PER_RUN_CSV)

summary_rows = []
if SPLIT_MODE in {'train_only', 'test_tune'}:
    summary_rows.append(
        {
            'metric': (
                'global_test_epoch_selection_pending'
                if SPLIT_MODE == 'test_tune'
                else 'no_evaluation'
            ),
            'mean': np.nan,
            'sd': np.nan,
            'n_runs': len(run_df),
            'evaluation_split': (
                'test_tuning' if SPLIT_MODE == 'test_tune' else 'none'
            ),
        }
    )
else:
    for metric_name in METRIC_NAMES:
        values = run_df[f'eval_{metric_name}'].to_numpy(dtype=float)
        summary_rows.append(
            {
                'metric': metric_name,
                'mean': float(np.mean(values)),
                'sd': (
                    float(np.std(values, ddof=1))
                    if len(values) > 1 else 0.0
                ),
                'n_runs': len(values),
                'evaluation_split': ','.join(
                    sorted(set(run_df['evaluation_split'].astype(str)))
                ),
            }
        )
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

logging.info("\n========== Selected evaluation summary ==========")
for summary_row in summary_rows:
    if summary_row['metric'] == 'no_evaluation':
        logging.info("  train_only: %d run(s), no evaluation", len(run_df))
    elif summary_row['metric'] == 'global_test_epoch_selection_pending':
        logging.info(
            "  test_tune: %d run(s), global epoch selection pending",
            len(run_df),
        )
    else:
        logging.info(
            "  %s: %.4f (%.4f)",
            summary_row['metric'],
            summary_row['mean'],
            summary_row['sd'],
        )
logging.info("Summary: %s", SUMMARY_CSV)
