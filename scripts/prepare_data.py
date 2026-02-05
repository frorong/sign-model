#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import h5py
from datasets.iam_ondb import load_iam_ondb
from datasets.preprocessing import (
    strokes_to_deltas,
    compute_statistics,
    standardize_strokes,
    pad_sequence,
    build_alphabet,
    filter_by_length,
)


def save_split(
    output_path: Path,
    samples: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    alphabet: str,
    max_len: int,
):
    all_strokes = []
    all_texts = []
    all_lengths = []
    
    for sample in samples:
        deltas = strokes_to_deltas(sample['strokes'])
        deltas = standardize_strokes(deltas, mean, std)
        padded, length = pad_sequence(deltas, max_len)
        
        all_strokes.append(padded)
        all_texts.append(sample['text'])
        all_lengths.append(length)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('strokes', data=np.array(all_strokes, dtype=np.float32))
        f.create_dataset('texts', data=np.array(all_texts, dtype=h5py.special_dtype(vlen=str)))
        f.create_dataset('lengths', data=np.array(all_lengths, dtype=np.int32))
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        f.attrs['alphabet'] = alphabet
        f.attrs['max_len'] = max_len


def main():
    parser = argparse.ArgumentParser(description='Prepare IAM-OnDB data')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to IAM-OnDB directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--max_len', type=int, default=700, help='Maximum sequence length')
    parser.add_argument('--min_len', type=int, default=10, help='Minimum sequence length')
    parser.add_argument('--train_split', type=float, default=0.9, help='Training split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading data from {data_dir}...')
    samples = load_iam_ondb(data_dir)
    print(f'Found {len(samples)} raw samples')
    
    print(f'Filtering by length ({args.min_len} - {args.max_len})...')
    samples = filter_by_length(samples, args.min_len, args.max_len)
    print(f'After filtering: {len(samples)} samples')
    
    samples_with_text = [s for s in samples if s['text']]
    print(f'Samples with text: {len(samples_with_text)}')
    
    if samples_with_text:
        samples = samples_with_text
    
    print('Computing global statistics...')
    all_deltas = [strokes_to_deltas(s['strokes']) for s in samples]
    mean, std = compute_statistics(all_deltas)
    print(f'Mean: {mean}, Std: {std}')
    
    print('Building alphabet...')
    alphabet = build_alphabet([s['text'] for s in samples])
    print(f'Alphabet ({len(alphabet)} chars): {repr(alphabet[:50])}...')
    
    print(f'Splitting data (train: {args.train_split}, val: {1 - args.train_split})...')
    np.random.seed(args.seed)
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * args.train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    print(f'Train samples: {len(train_samples)}, Val samples: {len(val_samples)}')
    
    train_path = output_dir / 'train.h5'
    val_path = output_dir / 'val.h5'
    
    print(f'Saving training data to {train_path}...')
    save_split(train_path, train_samples, mean, std, alphabet, args.max_len)
    
    print(f'Saving validation data to {val_path}...')
    save_split(val_path, val_samples, mean, std, alphabet, args.max_len)
    
    stats_path = output_dir / 'stats.npz'
    np.savez(stats_path, mean=mean, std=std, alphabet=np.array(list(alphabet)))
    print(f'Saved statistics to {stats_path}')
    
    print('Done!')
    print(f'\nDataset summary:')
    print(f'  Total samples: {len(samples)}')
    print(f'  Train samples: {len(train_samples)}')
    print(f'  Val samples: {len(val_samples)}')
    print(f'  Alphabet size: {len(alphabet)}')
    print(f'  Max sequence length: {args.max_len}')


if __name__ == '__main__':
    main()
