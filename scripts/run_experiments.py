#!/usr/bin/env python3
"""
Portable experiment runner for HAREN-CTC.

Runs the multi-seed training script on one GPU, sequentially, for a chosen
configuration. This is the simple, single-machine equivalent of the internal
multi-GPU queue used for the paper. Each configuration writes a log dir named
`logs_fixed_split_BCE_high_dropout_CTC005_3stra[_<RUN_TAG>]` next to where the
command is run, containing per-epoch / per-seed / summary CSVs.

Examples
--------
# Main result (paper seeds, LR 1e-5, class-balanced sampler):
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --run-tag main

# Learning-rate sweep with the sampler disabled (pre-balanced data):
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --lr 5e-6 --no-sampler --run-tag lr5e6_nos

# With synthetic data augmentation merged into the train set:
  python scripts/run_experiments.py \
      --data-root ./processed_data-utterance-fixed-split-nooffset \
      --cdoa-train-dir /path/to/synthetic_16k --run-tag aug199
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
                    help='Dir containing train/ val/ test/ (from preprocess.py).')
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
    ap.add_argument('--no-sampler', action='store_true',
                    help='Disable WeightedRandomSampler (use for pre-balanced data).')
    ap.add_argument('--cdoa-train-dir', default='',
                    help='Optional flat dir of synthetic train wavs to merge in.')
    ap.add_argument('--run-tag', default='',
                    help='Suffix for the output log dir name.')
    args = ap.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['DATA_ROOT'] = args.data_root
    env['SEEDS'] = args.seeds
    env['NUM_EPOCHS'] = str(args.epochs)
    env['BATCH_SIZE'] = str(args.batch_size)
    env['LR'] = str(args.lr)
    env['CDOA_TRAIN_DIR'] = args.cdoa_train_dir
    env['RUN_TAG'] = args.run_tag
    env['NO_SAMPLER'] = '1' if args.no_sampler else ''
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    cmd = [args.python, '-u', args.script]
    print('[run] ' + ' '.join(f'{k}={env[k]}' for k in
          ('CUDA_VISIBLE_DEVICES', 'DATA_ROOT', 'SEEDS', 'NUM_EPOCHS',
           'BATCH_SIZE', 'LR', 'NO_SAMPLER', 'CDOA_TRAIN_DIR', 'RUN_TAG')))
    print('[run] ' + ' '.join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == '__main__':
    main()
