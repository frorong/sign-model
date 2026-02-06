"""
여러 H5 데이터셋을 하나로 병합

사용법:
python scripts/merge_datasets.py \
    --inputs data/train.h5 data/iam_ondb.h5 \
    --output data/merged.h5 \
    --max_seq_len 700
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm


def load_h5_dataset(path: Path) -> dict:
    with h5py.File(path, 'r') as f:
        data = {
            'strokes': f['strokes'][:],
            'lengths': f['lengths'][:],
            'texts': [f['texts'][i] for i in range(len(f['texts']))],
            'alphabet': f.attrs.get('alphabet', ''),
        }
        
        if 'mean' in f:
            data['mean'] = f['mean'][:]
        if 'std' in f:
            data['std'] = f['std'][:]
    
    return data


def merge_datasets(datasets: list[dict], max_seq_len: int = 700) -> dict:
    all_alphabets = set()
    for d in datasets:
        all_alphabets.update(d['alphabet'])
    merged_alphabet = ''.join(sorted(all_alphabets))
    
    total_samples = sum(len(d['strokes']) for d in datasets)
    
    merged_strokes = np.zeros((total_samples, max_seq_len, 3), dtype=np.float32)
    merged_lengths = []
    merged_texts = []
    
    idx = 0
    for d in tqdm(datasets, desc='Merging'):
        for i in range(len(d['strokes'])):
            seq_len = min(d['lengths'][i], max_seq_len)
            merged_strokes[idx, :seq_len] = d['strokes'][i, :seq_len]
            merged_lengths.append(seq_len)
            merged_texts.append(d['texts'][i])
            idx += 1
    
    all_deltas = []
    for i in range(len(merged_strokes)):
        seq_len = merged_lengths[i]
        all_deltas.append(merged_strokes[i, :seq_len, :2])
    
    all_deltas = np.concatenate(all_deltas, axis=0)
    mean = all_deltas.mean(axis=0)
    std = all_deltas.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    
    return {
        'strokes': merged_strokes,
        'lengths': np.array(merged_lengths, dtype=np.int32),
        'texts': merged_texts,
        'alphabet': merged_alphabet,
        'mean': mean,
        'std': std,
    }


def save_merged(data: dict, output_path: Path, train_split: float = 0.9):
    n_samples = len(data['strokes'])
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_split)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    for split_name, split_idx in [('train', train_idx), ('val', val_idx)]:
        split_path = output_path.parent / f"{output_path.stem}_{split_name}.h5"
        
        with h5py.File(split_path, 'w') as f:
            f.create_dataset('strokes', data=data['strokes'][split_idx], compression='gzip')
            f.create_dataset('lengths', data=data['lengths'][split_idx])
            f.create_dataset('mean', data=data['mean'])
            f.create_dataset('std', data=data['std'])
            
            dt = h5py.special_dtype(vlen=str)
            texts_ds = f.create_dataset('texts', (len(split_idx),), dtype=dt)
            for i, idx in enumerate(split_idx):
                texts_ds[i] = data['texts'][idx]
            
            f.attrs['alphabet'] = data['alphabet']
            f.attrs['n_samples'] = len(split_idx)
        
        print(f"Saved {split_name}: {split_path} ({len(split_idx)} samples)")
    
    print(f"\nAlphabet size: {len(data['alphabet'])}")
    print(f"Total samples: {n_samples} (train: {n_train}, val: {n_samples - n_train})")


def main():
    parser = argparse.ArgumentParser(description='Merge multiple H5 datasets')
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help='Input H5 files to merge')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file prefix (will create _train.h5 and _val.h5)')
    parser.add_argument('--max_seq_len', type=int, default=700,
                        help='Maximum sequence length')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Training data ratio')
    
    args = parser.parse_args()
    
    print(f"Loading {len(args.inputs)} datasets...")
    datasets = [load_h5_dataset(Path(p)) for p in args.inputs]
    
    for i, (path, d) in enumerate(zip(args.inputs, datasets)):
        print(f"  [{i+1}] {path}: {len(d['strokes'])} samples, alphabet={len(d['alphabet'])}")
    
    print("\nMerging...")
    merged = merge_datasets(datasets, args.max_seq_len)
    
    print("\nSaving...")
    save_merged(merged, Path(args.output), args.train_split)


if __name__ == '__main__':
    main()
