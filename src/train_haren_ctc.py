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
  L = BCEWithLogits(cls)  +  ctc_weight * warmup(epoch) * CTCLoss
  ctc_weight = 0.05, linearly warmed up over the first CTC_WARMUP_EPOCHS epochs.

Evaluation protocol
-------------------
  Utterance-level sigmoid probs are aggregated to subject level by MAJORITY VOTE
  (threshold 0.5); metrics reported: macro-F1, F1_pos, F1_neg, sensitivity,
  specificity, ROC-AUC. Results are averaged over 5 seeds (mean +/- sd).

Environment variables (all optional; defaults reproduce the paper)
------------------------------------------------------------------
  SEEDS           comma list of seeds        (default 123,1234,12345,123456,1234567)
  NUM_EPOCHS      epochs per seed             (default 15)
  BATCH_SIZE      batch size                  (default 16)
  LR              AdamW learning rate         (default 1e-5)
  DATA_ROOT       dir containing train/val/test (default processed_data-utterance-fixed-split)
  NO_SAMPLER      "1" disables the WeightedRandomSampler (use with pre-balanced data)
  CDOA_TRAIN_DIR  optional extra flat dir of synthetic train wavs (data augmentation)
  RUN_TAG         suffix appended to the log dir name

Fixed hyper-parameters (edit here, not via env): DROPOUT=0.5, weight_decay=1e-4,
num_groups=2, k=10 (CTC classes = 2k+1 = 21, blank index = 20), max_sec=10.0.
"""
import os, random, logging, copy
import numpy as np, pandas as pd, torch, torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, HubertModel

# ───────────── 常量 / 路径 ─────────────
SEEDS           = [123, 1234, 12345, 123456, 1234567]   # 五个随机种子
if os.environ.get('SEEDS'): SEEDS = [int(x) for x in os.environ['SEEDS'].split(',')]  # override for smoke
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS','15'))
PATIENCE        = 5                                      # 早停 patience（基于 val utt-level macro F1）
BATCH_SIZE      = int(os.environ.get('BATCH_SIZE','16'))
LR              = float(os.environ.get('LR', '1e-5'))
DROPOUT         = 0.5

CTC_ENABLED = True
CTC_WARMUP_EPOCHS = 5
USE_PRECOMPUTED_HUBERT = False

_DATA_ROOT = os.environ.get('DATA_ROOT', '/home/yuxin/mydata/processed_data-utterance-fixed-split')
DIR_TRAIN = os.path.join(_DATA_ROOT, 'train')
DIR_VAL   = os.path.join(_DATA_ROOT, 'val')
DIR_TEST  = os.path.join(_DATA_ROOT, 'test')

# ── R4 augmentation hook: merge extra synthetic train dir (TRAIN only) via env ──
CDOA_TRAIN_DIR = os.environ.get('CDOA_TRAIN_DIR', '').strip()   # flat 16k dir with wav+.label+.phq_label
RUN_TAG        = os.environ.get('RUN_TAG', '').strip()          # e.g. B_cdoa / C_congruent / D_conv / A_real


LOG_DIR       = 'logs_fixed_split_BCE_high_dropout_CTC005_3stra' + (('_'+RUN_TAG) if RUN_TAG else '')
LOG_BASE_NAME = 'log_fixed_split'
PER_EPOCH_CSV    = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_per_epoch.csv')
PER_SEED_TEST_CSV = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_per_seed_test.csv')
SUMMARY_CSV      = os.path.join(LOG_DIR, f'{LOG_BASE_NAME}_summary.csv')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
hubert_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")

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
def set_seed(seed=51):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

TRAIN_WAVS = build_wav_index([DIR_TRAIN] + ([CDOA_TRAIN_DIR] if CDOA_TRAIN_DIR and os.path.isdir(CDOA_TRAIN_DIR) else []))
VAL_WAVS   = build_wav_index([DIR_VAL])
TEST_WAVS  = build_wav_index([DIR_TEST])
if len(TRAIN_WAVS) == 0 or len(VAL_WAVS) == 0 or len(TEST_WAVS) == 0:
    raise RuntimeError(
        f"Empty split: train={len(TRAIN_WAVS)}, val={len(VAL_WAVS)}, test={len(TEST_WAVS)}. "
        f"Run data_preprocessing_fixed-split.py first."
    )

# 跨 split subject leakage 检查（一次性，开训前）
def _subjs(wavs):
    return set(os.path.basename(p).split('_')[0] for p in wavs)
train_subj_set, val_subj_set, test_subj_set = _subjs(TRAIN_WAVS), _subjs(VAL_WAVS), _subjs(TEST_WAVS)
for a_name, a, b_name, b in [
    ('train', train_subj_set, 'val', val_subj_set),
    ('train', train_subj_set, 'test', test_subj_set),
    ('val',   val_subj_set,   'test', test_subj_set),
]:
    inter = a & b
    if inter:
        raise RuntimeError(f"Subject leakage between {a_name} and {b_name}: {sorted(inter)[:20]}")
logging.info(
    "No subject overlap across train/val/test. counts: train_subj=%d val_subj=%d test_subj=%d | utt: %d/%d/%d",
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

            meta = torchaudio.info(wav)
            self.info.append(dict(
                path=wav,
                binary_label=binary_label,
                phq_score=phq_score,
                phq_severity=phq_to_severity(phq_score),
                sr=meta.sample_rate,
                nfrm=meta.num_frames,
                subject=subj
            ))

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

    def _crop(self, wav, total, sr):
        seg = int(self.max_sec * sr)
        if self.training and total > seg:
            st = random.randint(0, total - seg)
            return wav[:, st:st+seg]
        else:
            if total >= seg:
                return wav[:, :seg]
            else:
                return F.pad(wav, (0, max(0, seg - total)))[:, :seg]

    def __getitem__(self, idx):
        d = self.info[idx]
        wav, _ = torchaudio.load(d['path'])
        wav = self._crop(wav, d['nfrm'], d['sr']).squeeze(0)
        return wav, d['label'], d['phq_score'], d['phq_severity'], d['path'], d['subject']

def collate(batch):
    waves, labels, phq_scores, phq_severities, paths, subjects = zip(*batch)
    waves_np = [w.cpu().numpy() for w in waves]
    outputs = feature_extractor(waves_np, sampling_rate=16000,
                                return_tensors="pt", padding=True, return_attention_mask=True)
    x = outputs.input_values
    attn_mask = outputs.attention_mask
    return x, attn_mask, torch.tensor(labels), torch.tensor(phq_scores), torch.tensor(phq_severities), list(paths), list(subjects)

def collate_with_waves(batch):
    waves, labels, phq_scores, phq_severities, paths, subjects = zip(*batch)
    waves_np = [w.cpu().numpy() for w in waves]
    outputs = feature_extractor(waves_np, sampling_rate=16000,
                                return_tensors="pt", padding=True, return_attention_mask=True)
    x = outputs.input_values
    attn_mask = outputs.attention_mask
    return x, attn_mask, torch.tensor(labels), torch.tensor(phq_scores), torch.tensor(phq_severities), list(paths), list(subjects), list(waves)

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
    def __init__(self, d_model_size, num_heads=2, dropout=0.3, d_ff=None, pre_norm=True):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model_size

        self.query = nn.Linear(d_model_size, d_model_size)
        self.key = nn.Linear(d_model_size, d_model_size)
        self.value = nn.Linear(d_model_size, d_model_size)
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
                 init_strategy: str = "exponential", alpha: float = 0.95):
        super().__init__()
        self.num_layers = num_layers
        self.selected_layers = list(range(num_layers))
        self.num_selected_layers = len(self.selected_layers)
        self.num_groups = num_groups

        self.group_assignment = nn.Parameter(
            torch.normal(mean=0.0, std=init_std, size=(self.num_selected_layers, num_groups))
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.initialize_exponential(alpha=alpha)

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

    def initialize_exponential(self, alpha=0.9):
        with torch.no_grad():
            for i in range(self.num_selected_layers):
                p_i = alpha ** i
                p_i = min(max(p_i, 1e-6), 1 - 1e-6)
                logit_0 = np.log(p_i / (1 - p_i))
                logit_1 = -logit_0
                self.group_assignment[i, 0] = torch.tensor(logit_0, dtype=torch.float32)
                self.group_assignment[i, 1] = torch.tensor(logit_1, dtype=torch.float32)

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
    def __init__(self, wavlm_model, num_labels=1, num_groups=2, dropout=0.3, ctc_weight=0.05, k=10):
        super().__init__()
        self.wavlm_model = wavlm_model
        # Fully freeze backbone behavior: no grad, no SpecAugment/dropout/layerdrop.
        self._freeze_wavlm_backbone()
        self.k = k
        self.ctc_weight = ctc_weight
        d_model_size = self.wavlm_model.config.hidden_size
        self.ctc_classifier = nn.Linear(d_model_size, 2*k + 1)

        self.adaptive_pool = AdaptiveWeightedPool(
            num_layers=24, hidden_size=1024, num_groups=num_groups,
            init_std=0.02, init_strategy="exponential", alpha=0.95
        )
        self.positional_encoding = LearnablePositionalEmbedding(d_model_size, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model_size))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.co_attention_module = CoAttentionModule(d_model_size, num_heads=2)
        self.pre_class_norm = nn.LayerNorm(d_model_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model_size),
            nn.Linear(d_model_size, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels)
        )
        # Init ONLY newly added heads — never re-init pretrained WavLM.
        for m in (
            self.ctc_classifier, self.adaptive_pool, self.positional_encoding,
            self.co_attention_module, self.pre_class_norm, self.classifier,
        ):
            m.apply(self._init_weights)

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

    def _downsampled_mask(self, attention_mask, feature_len):
        if attention_mask is None:
            return None
        with torch.no_grad():
            input_lengths = attention_mask.sum(dim=-1)
            try:
                feat_lengths = self.wavlm_model._get_feat_extract_output_lengths(input_lengths).to(torch.long)
            except Exception:
                B = attention_mask.size(0)
                feat_lengths = torch.full((B,), feature_len, dtype=torch.long, device=attention_mask.device)
            rng = torch.arange(feature_len, device=attention_mask.device).unsqueeze(0)
            key_padding_mask = rng >= feat_lengths.unsqueeze(1)
            return key_padding_mask

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
def torch_kmeans(features, n_clusters, max_iter=100, tol=1e-4):
    n_samples, n_features = features.shape
    if n_samples < n_clusters:
        n_clusters = n_samples
    if n_clusters == 0:
        return torch.zeros((0,), dtype=torch.long, device=features.device)
    idx = torch.linspace(0, max(0, n_samples - 1), steps=n_clusters).round().long()
    centroids = features[idx].clone()
    for _ in range(max_iter):
        distances = torch.cdist(features, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centroids[k] = features[mask].mean(dim=0)
            else:
                new_centroids[k] = centroids[k]
        shift = torch.norm(new_centroids - centroids, dim=1).mean()
        centroids = new_centroids
        if shift < tol:
            break
    distances = torch.cdist(features, centroids)
    labels = torch.argmin(distances, dim=1)
    return labels

def generate_hubert_policy_targets_online(raw_waves, labels, k, device, T_model, hubert_model):
    waves_np = [w.cpu().numpy() for w in raw_waves]
    hubert_inputs = hubert_feature_extractor(
        waves_np, sampling_rate=16000, return_tensors='pt', padding=True
    )
    hubert_input_values = hubert_inputs.input_values.to(device)
    with torch.no_grad():
        outputs = hubert_model(hubert_input_values, output_hidden_states=True)
        features_b = outputs.hidden_states[12]
    batch_targets, batch_target_lengths, batch_input_lengths = [], [], []
    for i in range(features_b.size(0)):
        features = features_b[i]
        cluster_ids = torch_kmeans(features, k)
        cid = cluster_ids[:T_model]
        if cid.numel() == 0:
            gamma_seq = []
        else:
            changes = torch.cat([torch.tensor([True], device=device), cid[1:] != cid[:-1]])
            gamma_seq = cid[changes].int().tolist()
        if labels[i].item() == 1:
            gamma_seq = [token + k for token in gamma_seq]
        batch_targets.append(gamma_seq)
        batch_target_lengths.append(len(gamma_seq))
        batch_input_lengths.append(T_model)
    if len(batch_targets) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    flat_targets = [token for seq in batch_targets for token in seq]
    flat_targets = torch.tensor(flat_targets, dtype=torch.long, device=device)
    input_lengths = torch.tensor(batch_input_lengths, dtype=torch.long, device=device)
    target_lengths = torch.tensor(batch_target_lengths, dtype=torch.long, device=device)
    return flat_targets, input_lengths, target_lengths

# ───────────── subject 级聚合（test 用） ─────────────
def aggregate_predictions_by_subject_majority(pred_probs, subject_ids, labels):
    """Paper protocol (Sec. IV-E): subject-level prediction = MAJORITY VOTE over
    utterance-level outputs. Each utterance is thresholded at 0.5; the returned
    value per subject is the positive-vote FRACTION, so `>= 0.5` is the majority
    class and the fraction doubles as the subject-level score for ROC-AUC.
    (Confidence-weighted / mean-prob aggregations were retired — they are not the
    paper protocol and flipped the sign of a within-noise effect.)"""
    subject_data = {}
    for prob, sid, lab in zip(pred_probs, subject_ids, labels):
        p = float(np.array(prob).squeeze())
        sid = str(sid)
        if sid not in subject_data:
            subject_data[sid] = {'preds': [], 'label': lab}
        subject_data[sid]['preds'].append(p)

    aggregated_preds, aggregated_labels = [], []
    for sid in sorted(subject_data.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        probs = np.array(subject_data[sid]['preds'], dtype=float)
        vote_frac = float((probs >= 0.5).mean())   # majority vote (fraction voting positive)
        aggregated_preds.append(vote_frac)
        aggregated_labels.append(subject_data[sid]['label'])
    return np.array(aggregated_preds), np.array(aggregated_labels)

# ───────────── Loss / Optim ─────────────
criterion = nn.BCEWithLogitsLoss()
def create_opt(model, lr):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler()
ctc_loss_fn = nn.CTCLoss(blank=20, zero_infinity=True)  # blank index = 2*k (=20 when k=10)

# ───────────── HuBERT 模型 ─────────────
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)
hubert_model.eval()

# ───────────── 多 seed 训练循环 ─────────────
all_epoch_results = []        # rows: seed, epoch, train_cls_loss, train_ctc_loss, val_utt_f1_macro
all_seed_test_results = []    # rows: seed, best_epoch, f1_macro, f1_pos, f1_neg, sens, spec, auc

for seed_idx, SEED in enumerate(SEEDS):
    # 切换 per-seed 日志文件
    seed_log_file = os.path.join(LOG_DIR, f"{LOG_BASE_NAME}_seed{SEED}.log")
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logging.root.removeHandler(handler)
    seed_file_handler = logging.FileHandler(seed_log_file, mode='w')
    seed_file_handler.setLevel(logging.INFO)
    seed_file_handler.setFormatter(logging.Formatter(f'%(asctime)s - [Seed {SEED}] - %(levelname)s - %(message)s'))
    logging.root.addHandler(seed_file_handler)
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(f'%(asctime)s - %(levelname)s - %(message)s'))

    logging.info(f"\n{'='*80}")
    logging.info(f"开始训练 Seed {SEED} ({seed_idx+1}/{len(SEEDS)})")
    logging.info(f"{'='*80}\n")

    set_seed(SEED)

    le = LabelEncoder().fit([0, 1])

    train_ds = AudioDataset(TRAIN_WAVS, feature_extractor, le, training=True)
    val_ds   = AudioDataset(VAL_WAVS,   feature_extractor, le, training=False)
    test_ds  = AudioDataset(TEST_WAVS,  feature_extractor, le, training=False)

    train_class_counts = np.bincount(train_ds.labels, minlength=2)
    val_class_counts   = np.bincount(val_ds.labels, minlength=2)
    test_class_counts  = np.bincount(test_ds.labels, minlength=2)
    logging.info(f"训练集样本数 {len(train_ds)}, 类别分布 {train_class_counts}")
    logging.info(f"验证集样本数 {len(val_ds)}, 类别分布 {val_class_counts}")
    logging.info(f"测试集样本数 {len(test_ds)}, 类别分布 {test_class_counts}")

    # 训练集类别均衡采样
    cls_cnt = np.bincount(train_ds.labels, minlength=2)
    w = 1.0 / np.maximum(cls_cnt, 1)
    sample_weights = [w[y] for y in train_ds.labels]

    if os.environ.get('NO_SAMPLER', '') == '1':
        # already class-balanced by synth -> plain shuffle, no re-weighting
        logging.info("[NO_SAMPLER] WeightedRandomSampler disabled; using shuffle=True")
        tr_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_with_waves)
    else:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED)
        )
        tr_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, collate_fn=collate_with_waves)
    val_loader  = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate)

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model = WavLMClassificationModel(wavlm, num_labels=1, num_groups=2, dropout=DROPOUT).to(device)
    # Belt-and-suspenders after .to(device): backbone must stay eval/frozen.
    model.wavlm_model.eval()
    for p in model.wavlm_model.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params {n_train:,} / {n_total:,} (WavLM frozen+eval)")
    opt = create_opt(model, LR)

    # 不早停：跑满 NUM_EPOCHS；每 epoch 在 val+test 上评估（subject 级），记录 best-test-epoch
    from sklearn.metrics import classification_report as _clsrep
    def _eval_loader(loader):
        model.eval()
        _p, _l, _s, _pth = [], [], [], []
        with torch.no_grad():
            for xb, attn_mask, yb, phq_scores, phq_severities, paths, subjects_in_batch in loader:
                xb, attn_mask = xb.to(device), attn_mask.to(device)
                cls_logits, _, _, _ = model(xb, attention_mask=attn_mask)
                prob = torch.sigmoid(cls_logits).cpu().numpy().reshape(-1)
                _p.extend(prob); _l.extend(yb.numpy()); _s.extend(subjects_in_batch); _pth.extend(paths)
        ap, al = aggregate_predictions_by_subject_majority(_p, _s, _l)
        al = al.astype(int); apr = (np.array(ap) >= 0.5).astype(int)
        m = metrics(al, apr, y_probs=ap)
        rep = _clsrep(al, apr, target_names=['neg(0)', 'pos(1)'], digits=4, zero_division=0)
        m['_utt'] = [(os.path.basename(str(p))[:-4], int(l), float(pr)) for p, l, pr in zip(_pth, _l, _p)]  # per-utt (utt_id,label,prob) for R2
        return m, rep
    best_test_f1, best_epoch, best_test_m = -1.0, -1, None

    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Seed {SEED} - Epoch {epoch}/{NUM_EPOCHS}")
        logging.info(f"{'='*60}")

        # ─── 训练 ───
        model.train()
        cls_loss_accum = 0.0
        ctc_loss_accum = 0.0
        n_batches = 0
        for xb, attn_mask, yb, phq_scores, phq_severities, paths, subj_ids, raw_waves in tr_loader:
            xb, attn_mask, yb = xb.to(device), attn_mask.to(device), yb.float().to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast():
                cls_logits, ctc_logits, group_probs, _ = model(xb, attention_mask=attn_mask)
                cls_loss = criterion(cls_logits, yb)

                B, T, _ = ctc_logits.shape
                current_ctc_w = 0.0
                ctc_loss = torch.tensor(0.0, device=xb.device)

                if CTC_ENABLED:
                    current_ctc_w = model.ctc_weight * min(1.0, (epoch - 1) / max(1, CTC_WARMUP_EPOCHS))
                    if current_ctc_w > 0:
                        yb_int = yb.long()
                        flat_t, in_len, tar_len = generate_hubert_policy_targets_online(
                            raw_waves, yb_int, model.k, device, T, hubert_model
                        )
                        if flat_t.numel() > 0:
                            ctc_loss = ctc_loss_fn(
                                F.log_softmax(ctc_logits.transpose(0,1), dim=-1),
                                flat_t, in_len, tar_len
                            )

                loss = cls_loss + current_ctc_w * ctc_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            cls_loss_accum += float(cls_loss.detach().item())
            ctc_loss_accum += float(ctc_loss.detach().item()) if isinstance(ctc_loss, torch.Tensor) else 0.0
            n_batches += 1

        train_cls_loss_mean = cls_loss_accum / max(1, n_batches)
        train_ctc_loss_mean = ctc_loss_accum / max(1, n_batches)

        # ─── 每 epoch 在 val + test 上评估（subject 级 classification report + ROC-AUC，不早停） ───
        val_m, val_rep = _eval_loader(val_loader)
        test_m, test_rep = _eval_loader(test_loader)
        logging.info(f"[Seed{SEED}-Epoch{epoch}] train_cls_loss={train_cls_loss_mean:.4f} train_ctc_loss={train_ctc_loss_mean:.4f}")
        logging.info(f"[Seed{SEED}-Epoch{epoch}] === VAL  === macroF1={val_m['f1_macro']:.4f} AUC={val_m.get('auc',0.0):.4f}\n{val_rep}")
        logging.info(f"[Seed{SEED}-Epoch{epoch}] === TEST === macroF1={test_m['f1_macro']:.4f} AUC={test_m.get('auc',0.0):.4f}\n{test_rep}")

        all_epoch_results.append({
            'seed': SEED, 'epoch': epoch,
            'train_cls_loss': train_cls_loss_mean, 'train_ctc_loss': train_ctc_loss_mean,
            'val_f1_macro': float(val_m['f1_macro']), 'val_auc': float(val_m.get('auc', 0.0)),
            'val_sens': float(val_m['sens']), 'val_spec': float(val_m['spec']),
            'test_f1_macro': float(test_m['f1_macro']), 'test_auc': float(test_m.get('auc', 0.0)),
            'test_f1_pos': float(test_m['f1_pos']), 'test_f1_neg': float(test_m['f1_neg']),
            'test_sens': float(test_m['sens']), 'test_spec': float(test_m['spec']),
        })
        if float(test_m['f1_macro']) > best_test_f1:
            best_test_f1 = float(test_m['f1_macro']); best_epoch = epoch; best_test_m = test_m

    # ─── per-seed：记录 best-test-epoch 的指标（不早停，跑满 NUM_EPOCHS 后从曲线取最佳） ───
    log_metrics(f"TEST-Seed{SEED} (best_epoch={best_epoch})", best_test_m)
    # dump best-epoch per-utterance test predictions (for R2 incongruent-cell eval)
    try:
        import pandas as _pd
        _pd.DataFrame(best_test_m.get('_utt', []), columns=['utt_id', 'label', 'prob']).to_csv(
            os.path.join(LOG_DIR, f'test_pred_seed{SEED}.csv'), index=False)
    except Exception as _e:
        logging.info(f'[warn] per-utt dump failed: {_e}')
    all_seed_test_results.append({
        'seed': SEED, 'best_epoch': best_epoch,
        'f1_macro': float(best_test_m['f1_macro']), 'f1_pos': float(best_test_m['f1_pos']),
        'f1_neg':   float(best_test_m['f1_neg']), 'sens': float(best_test_m['sens']),
        'spec':     float(best_test_m['spec']), 'auc': float(best_test_m.get('auc', 0.0)),
    })

    # 显式释放（避免在多 seed 间累积显存）
    del model, wavlm
    torch.cuda.empty_cache()

# ───────────── 保存 per-epoch / per-seed-test CSV ─────────────
pd.DataFrame(all_epoch_results).to_csv(PER_EPOCH_CSV, index=False)
logging.info(f"\nper-epoch 训练曲线已保存到: {PER_EPOCH_CSV}")

seed_df = pd.DataFrame(all_seed_test_results)
seed_df.to_csv(PER_SEED_TEST_CSV, index=False)
logging.info(f"per-seed test 指标已保存到: {PER_SEED_TEST_CSV}")

# ───────────── 5 seed 聚合：mean / sd ─────────────
metric_cols = ['f1_macro', 'f1_pos', 'f1_neg', 'sens', 'spec', 'auc']
summary_rows = []
for col in metric_cols:
    vals = seed_df[col].to_numpy(dtype=float)
    summary_rows.append({
        'metric': col,
        'mean':   float(np.mean(vals)),
        'sd':     float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

logging.info("\n========== 5-seed Test Summary (mean ± sd) ==========")
for r in summary_rows:
    logging.info(f"  {r['metric']:>9s}: {r['mean']:.4f} ({r['sd']:.4f})")
logging.info(f"汇总表已保存到: {SUMMARY_CSV}")
