#!/usr/bin/env python3
"""
DAIC-WOZ / AVEC-2017 utterance-level preprocessing for HAREN-CTC.

For every participant in an official AVEC-2017 split CSV this script:
  1. reads the participant's transcript CSV and keeps only 'Participant' turns,
  2. computes each turn's duration and sorts turns longest-first,
  3. applies the known per-subject timestamp corrections,
  4. drops turns shorter than --min-duration seconds,
  5. selects a number of turns depending on the split (see below),
  6. cuts the corresponding audio segment out of the full <PID>_AUDIO.wav and
     writes, for each selected turn, three files into the split output dir:
        <PID>_<n>.wav        the audio clip (original 16 kHz mono)
        <PID>_<n>.label      PHQ8 binary label (0 / 1)
        <PID>_<n>.phq_label  PHQ8 total score  (0 .. 24)

Turn-selection policy (matches the paper):
  * train split, sample_mode='proportional':
      non-depressed (label 0) -> 18 longest turns,
      depressed     (label 1) -> 46 longest turns   (mild class re-balancing).
  * val / test split, sample_mode='fixed':
      --fixed-n (default 20) longest turns per subject, regardless of label.

The four known transcript corrections are mandatory for this canonical
preprocessing path and are always applied.

Example
-------
  python preprocess.py \\
      --audio-dir  /path/to/DAIC/wav_files \\
      --trans-dir  /path/to/DAIC/transcripts \\
      --label-dir  /path/to/DAIC/labels \\
      --out-root   ./datasets/fixed_corrected_offset
"""
import os
import argparse
import pandas as pd
from pydub import AudioSegment

# Known transcript timestamp corrections (seconds) for the reported variant.
OFFSET_MAP = {
    "318": 34.0,
    "321": 3.355,
    "341": 6.07,
    "362": 16.54,
}


def load_one_split(csv_path):
    """Read one official split CSV -> Participant_ID / PHQ8_Binary / PHQ8_Score.

    No concat / drop_duplicates: a subject only ever appears in the split it
    officially belongs to. The full test CSV sometimes uses PHQ_* column names,
    so we defensively rename them to PHQ8_*.
    """
    df = pd.read_csv(csv_path)
    rename_map = {}
    if 'PHQ_Binary' in df.columns and 'PHQ8_Binary' not in df.columns:
        rename_map['PHQ_Binary'] = 'PHQ8_Binary'
    if 'PHQ_Score' in df.columns and 'PHQ8_Score' not in df.columns:
        rename_map['PHQ_Score'] = 'PHQ8_Score'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df[['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pick_utterances(transcript_csv, speaker='Participant'):
    """Return the speaker's turns with a 'duration' column, sorted longest-first."""
    df = pd.read_csv(transcript_csv)
    df = df[df['speaker'] == speaker].copy()
    if df.empty:
        return df
    df['duration'] = df['stop_time'].astype(float) - df['start_time'].astype(float)
    return df.sort_values('duration', ascending=False)


def process_subject(pid, label, phq_score, out_dir, audio_dir, trans_dir,
                    offset_map, sample_mode='proportional', fixed_n=20,
                    min_duration=1.0):
    """Cut and export clips for a single subject. Returns the number written."""
    audio_path = os.path.join(audio_dir, f'{pid}_AUDIO.wav')
    trans_path = os.path.join(trans_dir, f'{pid}_TRANSCRIPT.csv')

    if not os.path.exists(audio_path) or not os.path.exists(trans_path):
        print(f'[Skip] audio / transcript missing for {pid}')
        return 0

    utt_df = pick_utterances(trans_path)
    if utt_df.empty:
        print(f'[Skip] no participant utterances in {pid}')
        return 0

    offset = offset_map.get(str(pid), 0.0)
    if offset != 0.0:
        utt_df = utt_df.copy()
        utt_df['start_time'] = utt_df['start_time'].astype(float) + offset
        utt_df['stop_time'] = utt_df['stop_time'].astype(float) + offset
        utt_df['duration'] = utt_df['stop_time'] - utt_df['start_time']
        print(f'[Offset] {pid}: applied +{offset}s to transcript timestamps')

    utt_df = utt_df[utt_df['duration'] >= min_duration]
    if utt_df.empty:
        print(f'[Skip] no utterances longer than {min_duration}s for {pid}')
        return 0

    if sample_mode == 'proportional':
        n = 18 if label == 0 else 46
    else:  # fixed
        n = fixed_n
    utt_df = utt_df.head(n)

    audio = AudioSegment.from_wav(audio_path)
    cnt = 0
    for i, row in utt_df.iterrows():
        st_ms = int(row['start_time'] * 1000)
        ed_ms = int(row['stop_time'] * 1000)
        seg = audio[st_ms:ed_ms]

        if len(seg) < 1000:  # skip clips shorter than 1s
            print(f"Skipping {pid} clip {i}: segment length {len(seg)} ms < 1000 ms")
            continue

        cnt += 1
        seg.export(os.path.join(out_dir, f'{pid}_{cnt}.wav'), format='wav')
        with open(os.path.join(out_dir, f'{pid}_{cnt}.label'), 'w') as f:
            f.write(str(label))
        with open(os.path.join(out_dir, f'{pid}_{cnt}.phq_label'), 'w') as f:
            f.write(str(phq_score))
    return cnt


def run(out_dir, csv_path, audio_dir, trans_dir, offset_map,
        sample_mode, fixed_n=20, min_duration=1.0):
    ensure_dir(out_dir)
    df = load_one_split(csv_path)

    total = 0
    for _, row in df.iterrows():
        pid = int(row['Participant_ID'])
        label = int(row['PHQ8_Binary'])
        phq_score = int(row['PHQ8_Score'])
        n_ok = process_subject(pid, label, phq_score, out_dir, audio_dir,
                               trans_dir, offset_map, sample_mode=sample_mode,
                               fixed_n=fixed_n, min_duration=min_duration)
        total += n_ok
        print(f'[{sample_mode}] {pid} (Binary={label}, PHQ={phq_score}): {n_ok} clips')
    print(f'\nDone!  Total clips in {out_dir} ({sample_mode}) = {total}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio-dir', required=True,
                    help='Dir with <PID>_AUDIO.wav files.')
    ap.add_argument('--trans-dir', required=True,
                    help='Dir with <PID>_TRANSCRIPT.csv files.')
    ap.add_argument('--label-dir', required=True,
                    help='Dir containing the official split CSVs.')
    ap.add_argument('--out-root', required=True,
                    help='Output root; train/ val/ test/ are created under it.')
    ap.add_argument('--csv-train', default='train_split_Depression_AVEC2017.csv')
    ap.add_argument('--csv-val',   default='dev_split_Depression_AVEC2017.csv')
    ap.add_argument('--csv-test',  default='full_test_split.csv')
    ap.add_argument('--fixed-n', type=int, default=20,
                    help='Turns per subject for val/test (default 20).')
    ap.add_argument('--min-duration', type=float, default=1.0,
                    help='Drop turns shorter than this many seconds (default 1.0).')
    args = ap.parse_args()

    run(os.path.join(args.out_root, 'train'),
        os.path.join(args.label_dir, args.csv_train),
        args.audio_dir, args.trans_dir, OFFSET_MAP,
        sample_mode='proportional', min_duration=args.min_duration)

    run(os.path.join(args.out_root, 'val'),
        os.path.join(args.label_dir, args.csv_val),
        args.audio_dir, args.trans_dir, OFFSET_MAP,
        sample_mode='fixed', fixed_n=args.fixed_n, min_duration=args.min_duration)

    run(os.path.join(args.out_root, 'test'),
        os.path.join(args.label_dir, args.csv_test),
        args.audio_dir, args.trans_dir, OFFSET_MAP,
        sample_mode='fixed', fixed_n=args.fixed_n, min_duration=args.min_duration)


if __name__ == '__main__':
    main()
