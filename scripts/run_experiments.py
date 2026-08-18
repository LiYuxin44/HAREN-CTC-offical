#!/usr/bin/env python3
"""
Portable experiment runner for HAREN-CTC.

Runs the multi-seed training script on one GPU, sequentially, for one explicit
configuration. Use `--output-dir` to isolate parallel or publication runs.

Examples
--------
# Dev-only hyperparameter trial:
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --lr 5e-5 --weight-decay 1e-4 --dropout 0.3 \
      --ctc-enabled 1 --test-policy none --run-tag hpo_trial

# One-time test evaluation of a frozen dev-selected checkpoint:
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --seeds 123 --test-policy final_only \
      --eval-checkpoint /path/to/seed123_best_dev.pt --run-tag final_test
"""
import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.normpath(os.path.join(HERE, '..', 'src', 'train_haren_ctc.py'))
PAPER_SEEDS = "123,1234,12345,123456,1234567"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', required=True,
                    help='Fixed-split root or PHQ CV audio source root.')
    ap.add_argument('--script', default=DEFAULT_SCRIPT,
                    help='Path to train_haren_ctc.py.')
    ap.add_argument('--python', default=sys.executable,
                    help='Python interpreter to use (default: current).')
    ap.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value.')
    ap.add_argument('--seeds', default=PAPER_SEEDS,
                    help='Comma-separated seeds (default: the 5 paper seeds).')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', default='1e-5')
    ap.add_argument('--weight-decay', default='1e-4')
    ap.add_argument('--dropout', default='0.3')
    ap.add_argument('--ctc-enabled', type=int, choices=(0, 1), default=1)
    ap.add_argument('--ctc-weight', default='0.05')
    ap.add_argument(
        '--ctc-mode',
        choices=('fixed', 'adaptive_ratio', 'shared_grad_norm'),
        default='fixed',
    )
    ap.add_argument(
        '--ctc-target-mode',
        choices=('label_shifted', 'neutral'),
        default='label_shifted',
    )
    ap.add_argument('--ctc-target-ratio', default='0.1')
    ap.add_argument('--ctc-warmup-epochs', type=int, default=5)
    ap.add_argument('--ctc-k', type=int, default=10)
    ap.add_argument('--ctc-grad-target-ratio', default='0.1')
    ap.add_argument('--ctc-grad-update-interval', type=int, default=10)
    ap.add_argument('--ctc-grad-ema-decay', default='0.9')
    ap.add_argument('--ctc-warmup-ratio', default='0.1')
    ap.add_argument(
        '--ctc-loss-policy',
        choices=('legacy_mean', 'normalized_fp32'),
        default='legacy_mean',
    )
    ap.add_argument(
        '--temporal-head-policy',
        choices=('legacy_2k1', 'neutral_k1'),
        default='legacy_2k1',
    )
    ap.add_argument('--global-unit-stride', type=int, default=1)
    ap.add_argument('--global-unit-reverse', type=int, choices=(0, 1), default=0)
    ap.add_argument('--crop-alignment-samples', type=int, default=1)
    ap.add_argument(
        '--wavlm-mask-policy',
        choices=('legacy_full', 'true_length'),
        default='legacy_full',
    )
    ap.add_argument(
        '--wavlm-preprocess-policy',
        choices=('legacy_prepad', 'valid_then_pad'),
        default='legacy_prepad',
    )
    ap.add_argument(
        '--wavlm-batch-padding-policy',
        choices=('fixed_10s', 'longest'),
        default='fixed_10s',
    )
    ap.add_argument(
        '--train-crop-policy',
        choices=('worker_random', 'epoch_keyed'),
        default='worker_random',
    )
    ap.add_argument(
        '--optimizer-policy',
        choices=('legacy_adamw', 'no_decay_warmup_cosine'),
        default='legacy_adamw',
    )
    ap.add_argument(
        '--head-arch-policy',
        choices=('legacy_17m', 'compact_9m'),
        default='legacy_17m',
    )
    ap.add_argument(
        '--head-init-policy',
        choices=('legacy_stream', 'component_seeded_v1'),
        default='legacy_stream',
    )
    ap.add_argument('--lr-warmup-ratio', default='0.1')
    ap.add_argument('--lr-min-ratio', default='0.1')
    ap.add_argument('--ema-decay', default='0')
    ap.add_argument('--save-training-state', type=int, choices=(0, 1), default=0)
    ap.add_argument('--resume-checkpoint', default='')
    ap.add_argument(
        '--routing-init-policy',
        choices=('legacy', 'probability_correct'),
        default='legacy',
    )
    ap.add_argument(
        '--sampling-policy',
        choices=('utterance_class_balanced', 'subject_class_balanced'),
        default='utterance_class_balanced',
    )
    ap.add_argument(
        '--aggregation-policy',
        choices=(
            'mean_probability',
            'confidence_weighted_probability',
            'confidence_weighted_vote',
        ),
        default='mean_probability',
    )
    ap.add_argument(
        '--temporal-target-policy',
        choices=(
            'local_kmeans_ctc',
            'global_units_ctc',
            'global_units_frame_ce',
        ),
        default='local_kmeans_ctc',
    )
    ap.add_argument('--global-unit-cache', default='')
    ap.add_argument('--grad-diagnostic-interval', type=int, default=0)
    ap.add_argument('--grad-clip-norm', default='0')
    ap.add_argument(
        '--grad-clip-policy',
        choices=('global', 'task_grouped'),
        default='global',
    )
    ap.add_argument('--silence-threshold-dbfs', default='-40')
    ap.add_argument(
        '--amp-dtype', choices=('fp16', 'bf16', 'fp32'), default='fp16'
    )
    ap.add_argument(
        '--eval-crop-policy',
        choices=('head', 'center', 'tail', 'multi3'),
        default='head',
    )
    ap.add_argument('--eval-window-stride-seconds', type=float, default=5.0)
    ap.add_argument(
        '--test-policy',
        choices=('none', 'final_only'),
        default='none',
        help='Training uses none; final_only requires a frozen checkpoint.',
    )
    ap.add_argument(
        '--eval-checkpoint',
        default='',
        help='Frozen checkpoint for one-time official-test evaluation.',
    )
    ap.add_argument('--run-tag', default='',
                    help='Suffix for the output log dir name.')
    ap.add_argument('--output-dir', default='',
                    help='Exact output directory (recommended for parallel jobs).')
    ap.add_argument(
        '--split-mode',
        choices=('fixed', 'test_tune'),
        default='fixed',
    )
    ap.add_argument('--train-manifest', default='',
                    help='PHQ CV train+dev manifest.')
    ap.add_argument('--val-manifest', default='',
                    help='PHQ CV held-out-fold manifest.')
    ap.add_argument('--fold-index', type=int, default=-1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--prefetch-factor', type=int, default=2)
    ap.add_argument('--log-interval', type=int, default=25)
    args = ap.parse_args()
    if args.ctc_k <= 0:
        ap.error('--ctc-k must be positive')
    if args.ctc_grad_update_interval <= 0:
        ap.error('--ctc-grad-update-interval must be positive')
    if args.global_unit_stride <= 0:
        ap.error('--global-unit-stride must be positive')
    if args.crop_alignment_samples <= 0:
        ap.error('--crop-alignment-samples must be positive')
    if args.grad_diagnostic_interval < 0 or float(args.grad_clip_norm) < 0:
        ap.error('gradient diagnostic interval and clip norm must be nonnegative')
    if not 0 < args.eval_window_stride_seconds <= 10:
        ap.error('--eval-window-stride-seconds must be in (0, 10]')
    if (
        args.ctc_enabled
        and args.temporal_target_policy.startswith('global_units_')
        and not args.global_unit_cache
    ):
        ap.error(
            '--global-unit-cache is required for global temporal targets'
        )
    if args.split_mode == 'test_tune' and not (
        args.train_manifest and args.val_manifest
    ):
        ap.error(
            '--split-mode test_tune requires --train-manifest and --val-manifest'
        )
    if (
        args.split_mode == 'test_tune'
        and args.fold_index not in range(5)
    ):
        ap.error('--split-mode test_tune requires --fold-index in 0..4')
    if (args.test_policy == 'final_only') != bool(args.eval_checkpoint):
        ap.error(
            '--test-policy final_only requires --eval-checkpoint, and '
            '--eval-checkpoint is forbidden with --test-policy none'
        )
    if args.split_mode == 'test_tune' and args.test_policy != 'none':
        ap.error(
            'test_tune mode does not expose the official fixed test split'
        )
    if args.split_mode == 'fixed' and (
        args.train_manifest or args.val_manifest or args.fold_index != -1
    ):
        ap.error(
            'fixed mode forbids manifests and requires --fold-index -1'
        )

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['DATA_ROOT'] = args.data_root
    env['SEEDS'] = args.seeds
    env['NUM_EPOCHS'] = str(args.epochs)
    env['BATCH_SIZE'] = str(args.batch_size)
    env['LR'] = str(args.lr)
    env['WEIGHT_DECAY'] = str(args.weight_decay)
    env['DROPOUT'] = str(args.dropout)
    env['CTC_ENABLED'] = str(args.ctc_enabled)
    env['CTC_WEIGHT'] = str(args.ctc_weight)
    env['CTC_MODE'] = args.ctc_mode
    env['CTC_TARGET_MODE'] = args.ctc_target_mode
    env['CTC_TARGET_RATIO'] = str(args.ctc_target_ratio)
    env['CTC_WARMUP_EPOCHS'] = str(args.ctc_warmup_epochs)
    env['CTC_K'] = str(args.ctc_k)
    env['CTC_GRAD_TARGET_RATIO'] = str(args.ctc_grad_target_ratio)
    env['CTC_GRAD_UPDATE_INTERVAL'] = str(args.ctc_grad_update_interval)
    env['CTC_GRAD_EMA_DECAY'] = str(args.ctc_grad_ema_decay)
    env['CTC_WARMUP_RATIO'] = str(args.ctc_warmup_ratio)
    env['CTC_LOSS_POLICY'] = args.ctc_loss_policy
    env['TEMPORAL_HEAD_POLICY'] = args.temporal_head_policy
    env['GLOBAL_UNIT_STRIDE'] = str(args.global_unit_stride)
    env['GLOBAL_UNIT_REVERSE'] = str(args.global_unit_reverse)
    env['CROP_ALIGNMENT_SAMPLES'] = str(args.crop_alignment_samples)
    env['WAVLM_MASK_POLICY'] = args.wavlm_mask_policy
    env['WAVLM_PREPROCESS_POLICY'] = args.wavlm_preprocess_policy
    env['WAVLM_BATCH_PADDING_POLICY'] = args.wavlm_batch_padding_policy
    env['TRAIN_CROP_POLICY'] = args.train_crop_policy
    env['OPTIMIZER_POLICY'] = args.optimizer_policy
    env['HEAD_ARCH_POLICY'] = args.head_arch_policy
    env['HEAD_INIT_POLICY'] = args.head_init_policy
    env['LR_WARMUP_RATIO'] = str(args.lr_warmup_ratio)
    env['LR_MIN_RATIO'] = str(args.lr_min_ratio)
    env['EMA_DECAY'] = str(args.ema_decay)
    env['SAVE_TRAINING_STATE'] = str(args.save_training_state)
    env['RESUME_CHECKPOINT'] = args.resume_checkpoint
    env['ROUTING_INIT_POLICY'] = args.routing_init_policy
    env['SAMPLING_POLICY'] = args.sampling_policy
    env['AGGREGATION_POLICY'] = args.aggregation_policy
    env['TEMPORAL_TARGET_POLICY'] = args.temporal_target_policy
    env['GLOBAL_UNIT_CACHE'] = args.global_unit_cache
    env['GRAD_DIAGNOSTIC_INTERVAL'] = str(
        args.grad_diagnostic_interval
    )
    env['GRAD_CLIP_NORM'] = str(args.grad_clip_norm)
    env['GRAD_CLIP_POLICY'] = args.grad_clip_policy
    env['SILENCE_THRESHOLD_DBFS'] = str(args.silence_threshold_dbfs)
    env['AMP_DTYPE'] = args.amp_dtype
    env['EVAL_CROP_POLICY'] = args.eval_crop_policy
    env['EVAL_WINDOW_STRIDE_SECONDS'] = str(
        args.eval_window_stride_seconds
    )
    env['TEST_POLICY'] = args.test_policy
    env['EVAL_CHECKPOINT'] = args.eval_checkpoint
    env['TEMPORAL_EVAL_DIAGNOSTICS'] = '0'
    env['RUN_TAG'] = args.run_tag
    env['OUTPUT_DIR'] = args.output_dir
    env['SPLIT_MODE'] = args.split_mode
    env['TRAIN_MANIFEST'] = args.train_manifest
    env['VAL_MANIFEST'] = args.val_manifest
    env['FOLD_INDEX'] = str(args.fold_index)
    env['NUM_WORKERS'] = str(args.workers)
    env['PREFETCH_FACTOR'] = str(args.prefetch_factor)
    env['LOG_INTERVAL'] = str(args.log_interval)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    cmd = [args.python, '-u', args.script]
    print('[run] ' + ' '.join(f'{k}={env[k]}' for k in
          ('CUDA_VISIBLE_DEVICES', 'DATA_ROOT', 'SEEDS', 'NUM_EPOCHS',
           'BATCH_SIZE', 'LR', 'WEIGHT_DECAY', 'DROPOUT', 'CTC_ENABLED',
           'CTC_WEIGHT', 'CTC_MODE', 'CTC_TARGET_MODE', 'CTC_TARGET_RATIO',
           'CTC_K', 'CTC_GRAD_TARGET_RATIO',
           'CTC_GRAD_UPDATE_INTERVAL', 'CTC_GRAD_EMA_DECAY',
           'CTC_WARMUP_RATIO', 'CTC_LOSS_POLICY',
           'TEMPORAL_HEAD_POLICY', 'GLOBAL_UNIT_STRIDE',
           'GLOBAL_UNIT_REVERSE', 'CROP_ALIGNMENT_SAMPLES',
           'WAVLM_MASK_POLICY', 'ROUTING_INIT_POLICY',
           'WAVLM_PREPROCESS_POLICY', 'WAVLM_BATCH_PADDING_POLICY',
           'TRAIN_CROP_POLICY',
           'OPTIMIZER_POLICY', 'LR_WARMUP_RATIO', 'LR_MIN_RATIO',
           'HEAD_ARCH_POLICY', 'HEAD_INIT_POLICY',
           'EMA_DECAY', 'SAVE_TRAINING_STATE',
           'SAMPLING_POLICY', 'AGGREGATION_POLICY',
           'TEMPORAL_TARGET_POLICY', 'GRAD_DIAGNOSTIC_INTERVAL',
           'GRAD_CLIP_NORM', 'GRAD_CLIP_POLICY', 'AMP_DTYPE',
           'EVAL_CROP_POLICY',
           'EVAL_WINDOW_STRIDE_SECONDS',
           'TEST_POLICY', 'SPLIT_MODE', 'FOLD_INDEX',
           'NUM_WORKERS', 'PREFETCH_FACTOR', 'RUN_TAG',
           'OUTPUT_DIR')))
    if args.eval_checkpoint:
        print(f"[run] EVAL_CHECKPOINT={env['EVAL_CHECKPOINT']}")
    if args.resume_checkpoint:
        print(f"[run] RESUME_CHECKPOINT={env['RESUME_CHECKPOINT']}")
    if args.split_mode == 'test_tune':
        print(
            '[run] '
            f"TRAIN_MANIFEST={env['TRAIN_MANIFEST']} "
            f"VAL_MANIFEST={env['VAL_MANIFEST']}"
        )
    print('[run] ' + ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
