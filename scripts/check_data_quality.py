#!/usr/bin/env python3
"""데이터 품질 점검 스크립트"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def analyze_dataset(h5_path: str) -> dict:
    """데이터셋 분석"""
    with h5py.File(h5_path, 'r') as f:
        strokes = f['strokes'][:]
        texts = [t.decode('utf-8') if isinstance(t, bytes) else t for t in f['texts'][:]]
        stroke_lengths = f['lengths'][:]
        text_lengths = np.array([len(t) for t in texts])
        alphabet = f.attrs['alphabet']
    
    n_samples = len(texts)
    
    stats = {
        'n_samples': n_samples,
        'alphabet_size': len(alphabet),
        'alphabet': alphabet,
    }
    
    stats['stroke_length'] = {
        'min': int(stroke_lengths.min()),
        'max': int(stroke_lengths.max()),
        'mean': float(stroke_lengths.mean()),
        'std': float(stroke_lengths.std()),
        'median': float(np.median(stroke_lengths)),
    }
    
    stats['text_length'] = {
        'min': int(text_lengths.min()),
        'max': int(text_lengths.max()),
        'mean': float(text_lengths.mean()),
        'std': float(text_lengths.std()),
    }
    
    steps_per_char = stroke_lengths / np.maximum(text_lengths, 1)
    stats['steps_per_char'] = {
        'min': float(steps_per_char.min()),
        'max': float(steps_per_char.max()),
        'mean': float(steps_per_char.mean()),
        'std': float(steps_per_char.std()),
    }
    
    return stats, strokes, texts, stroke_lengths, text_lengths


def find_anomalies(strokes: np.ndarray, texts: list, stroke_lengths: np.ndarray, 
                   text_lengths: np.ndarray) -> dict:
    """이상치 탐지"""
    n_samples = len(texts)
    anomalies = {
        'too_short': [],
        'too_long': [],
        'extreme_steps_per_char': [],
        'large_jumps': [],
        'static_strokes': [],
        'extreme_pen_up': [],
    }
    
    steps_per_char = stroke_lengths / np.maximum(text_lengths, 1)
    spc_mean = steps_per_char.mean()
    spc_std = steps_per_char.std()
    
    for i in range(n_samples):
        length = stroke_lengths[i]
        text = texts[i]
        spc = steps_per_char[i]
        
        if length < 10:
            anomalies['too_short'].append((i, text, length))
        
        if length > 600:
            anomalies['too_long'].append((i, text, length))
        
        if abs(spc - spc_mean) > 3 * spc_std:
            anomalies['extreme_steps_per_char'].append((i, text, spc))
        
        sample_strokes = strokes[i, :length]
        deltas = sample_strokes[:, :2]
        step_sizes = np.linalg.norm(deltas, axis=1)
        
        if step_sizes.max() > 5.0:
            anomalies['large_jumps'].append((i, text, float(step_sizes.max())))
        
        if step_sizes.mean() < 0.01:
            anomalies['static_strokes'].append((i, text, float(step_sizes.mean())))
        
        pen_ups = sample_strokes[:, 2]
        pen_up_ratio = pen_ups.mean()
        if pen_up_ratio > 0.5 or pen_up_ratio < 0.01:
            anomalies['extreme_pen_up'].append((i, text, float(pen_up_ratio)))
    
    return anomalies


def visualize_samples(strokes: np.ndarray, texts: list, stroke_lengths: np.ndarray,
                      indices: list, output_path: str, title: str = "Samples"):
    """샘플 시각화"""
    n = min(len(indices), 12)
    if n == 0:
        return
    
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for ax_idx, i in enumerate(indices[:n]):
        ax = axes[ax_idx]
        length = stroke_lengths[i]
        sample = strokes[i, :length]
        
        coords = np.cumsum(sample[:, :2], axis=0)
        pen_ups = sample[:, 2] > 0.5
        
        start = 0
        for j in range(1, length):
            if pen_ups[j-1]:
                if j - start > 1:
                    ax.plot(coords[start:j, 0], -coords[start:j, 1], 'b-', linewidth=1)
                start = j
        if length - start > 1:
            ax.plot(coords[start:, 0], -coords[start:, 1], 'b-', linewidth=1)
        
        ax.set_title(f"[{i}] {texts[i][:30]}", fontsize=8)
        ax.set_aspect('equal')
        ax.axis('off')
    
    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_distributions(stats: dict, stroke_lengths: np.ndarray, text_lengths: np.ndarray,
                       output_path: str):
    """분포 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(stroke_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(stats['stroke_length']['mean'], color='r', linestyle='--', label=f"Mean: {stats['stroke_length']['mean']:.1f}")
    axes[0, 0].set_xlabel('Stroke Length')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Stroke Length Distribution')
    axes[0, 0].legend()
    
    axes[0, 1].hist(text_lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(stats['text_length']['mean'], color='r', linestyle='--', label=f"Mean: {stats['text_length']['mean']:.1f}")
    axes[0, 1].set_xlabel('Text Length')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Text Length Distribution')
    axes[0, 1].legend()
    
    steps_per_char = stroke_lengths / np.maximum(text_lengths, 1)
    axes[1, 0].hist(steps_per_char, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(stats['steps_per_char']['mean'], color='r', linestyle='--', label=f"Mean: {stats['steps_per_char']['mean']:.1f}")
    axes[1, 0].set_xlabel('Steps per Character')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Steps per Character Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(text_lengths, stroke_lengths, alpha=0.3, s=5)
    axes[1, 1].set_xlabel('Text Length')
    axes[1, 1].set_ylabel('Stroke Length')
    axes[1, 1].set_title('Text Length vs Stroke Length')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Check data quality')
    parser.add_argument('--data', type=str, default='data/train.h5', help='Path to H5 data file')
    parser.add_argument('--output_dir', type=str, default='data_quality')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {args.data}...")
    stats, strokes, texts, stroke_lengths, text_lengths = analyze_dataset(args.data)
    
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Samples: {stats['n_samples']}")
    print(f"Alphabet size: {stats['alphabet_size']}")
    print(f"\nStroke Length:")
    print(f"  Min: {stats['stroke_length']['min']}, Max: {stats['stroke_length']['max']}")
    print(f"  Mean: {stats['stroke_length']['mean']:.1f} ± {stats['stroke_length']['std']:.1f}")
    print(f"\nText Length:")
    print(f"  Min: {stats['text_length']['min']}, Max: {stats['text_length']['max']}")
    print(f"  Mean: {stats['text_length']['mean']:.1f} ± {stats['text_length']['std']:.1f}")
    print(f"\nSteps per Character:")
    print(f"  Min: {stats['steps_per_char']['min']:.1f}, Max: {stats['steps_per_char']['max']:.1f}")
    print(f"  Mean: {stats['steps_per_char']['mean']:.1f} ± {stats['steps_per_char']['std']:.1f}")
    
    print("\n" + "=" * 50)
    print("ANOMALY DETECTION")
    print("=" * 50)
    anomalies = find_anomalies(strokes, texts, stroke_lengths, text_lengths)
    
    for anomaly_type, items in anomalies.items():
        print(f"\n{anomaly_type}: {len(items)} samples")
        if items and len(items) <= 5:
            for item in items[:5]:
                print(f"  [{item[0]}] '{item[1][:30]}' - {item[2]}")
        elif items:
            print(f"  (showing first 5)")
            for item in items[:5]:
                print(f"  [{item[0]}] '{item[1][:30]}' - {item[2]}")
    
    if args.visualize:
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)
        
        plot_distributions(stats, stroke_lengths, text_lengths, 
                          str(output_dir / 'distributions.png'))
        
        random_indices = np.random.choice(len(texts), min(12, len(texts)), replace=False)
        visualize_samples(strokes, texts, stroke_lengths, list(random_indices),
                         str(output_dir / 'random_samples.png'), "Random Samples")
        
        if anomalies['large_jumps']:
            indices = [x[0] for x in anomalies['large_jumps'][:12]]
            visualize_samples(strokes, texts, stroke_lengths, indices,
                             str(output_dir / 'anomaly_large_jumps.png'), "Anomaly: Large Jumps")
        
        if anomalies['extreme_steps_per_char']:
            indices = [x[0] for x in anomalies['extreme_steps_per_char'][:12]]
            visualize_samples(strokes, texts, stroke_lengths, indices,
                             str(output_dir / 'anomaly_extreme_spc.png'), "Anomaly: Extreme Steps/Char")
    
    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"Dataset: `{args.data}`\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Samples: {stats['n_samples']}\n")
        f.write(f"- Alphabet size: {stats['alphabet_size']}\n\n")
        f.write("### Stroke Length\n")
        f.write(f"- Range: [{stats['stroke_length']['min']}, {stats['stroke_length']['max']}]\n")
        f.write(f"- Mean: {stats['stroke_length']['mean']:.1f} ± {stats['stroke_length']['std']:.1f}\n\n")
        f.write("### Text Length\n")
        f.write(f"- Range: [{stats['text_length']['min']}, {stats['text_length']['max']}]\n")
        f.write(f"- Mean: {stats['text_length']['mean']:.1f} ± {stats['text_length']['std']:.1f}\n\n")
        f.write("## Anomalies\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for anomaly_type, items in anomalies.items():
            f.write(f"| {anomaly_type} | {len(items)} |\n")
    
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
