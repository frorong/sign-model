#!/usr/bin/env python3
"""모델 평가 및 비교 스크립트"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import yaml
import numpy as np
from datetime import datetime
from models import SynthesisNetwork
from inference.sampler import Sampler, PRESETS


EVAL_TEXTS = [
    "hello",
    "world",
    "The quick brown fox jumps",
    "SIGNATURE",
    "abcdefghijklmnopqrstuvwxyz",
]


def load_model(model_path: Path, config: dict, alphabet: str, device: torch.device) -> SynthesisNetwork:
    model = SynthesisNetwork(
        alphabet_size=len(alphabet),
        hidden_size=config['model']['hidden_size'],
        num_mixtures=config['model']['num_mixtures'],
        num_attention_components=config['model']['num_attention_components']
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def compute_stroke_stats(strokes: np.ndarray) -> dict:
    """스트로크 통계 계산"""
    coords = np.cumsum(strokes[:, :2], axis=0)
    pen_ups = strokes[:, 2] > 0.5
    
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    
    step_sizes = np.linalg.norm(strokes[1:, :2], axis=1)
    
    return {
        'length': len(strokes),
        'x_range': float(x_range),
        'y_range': float(y_range),
        'aspect_ratio': float(x_range / max(y_range, 1e-6)),
        'pen_up_count': int(pen_ups.sum()),
        'pen_up_ratio': float(pen_ups.mean()),
        'avg_step': float(step_sizes.mean()),
        'max_step': float(step_sizes.max()),
        'std_step': float(step_sizes.std()),
    }


def evaluate_model(model_path: Path, config: dict, alphabet: str, device: torch.device) -> dict:
    """단일 모델 평가"""
    model = load_model(model_path, config, alphabet, device)
    sampler = Sampler(model, alphabet, device)
    
    results = {
        'model': str(model_path),
        'texts': {},
    }
    
    for preset_name in ['normal', 'sharp', 'loose']:
        preset_results = []
        
        for text in EVAL_TEXTS:
            strokes = sampler.generate_with_preset(text, preset=preset_name)
            stats = compute_stroke_stats(strokes)
            stats['text'] = text
            stats['text_len'] = len(text)
            stats['steps_per_char'] = stats['length'] / len(text)
            preset_results.append(stats)
        
        avg_stats = {
            'avg_length': np.mean([r['length'] for r in preset_results]),
            'avg_steps_per_char': np.mean([r['steps_per_char'] for r in preset_results]),
            'avg_pen_up_ratio': np.mean([r['pen_up_ratio'] for r in preset_results]),
            'avg_aspect_ratio': np.mean([r['aspect_ratio'] for r in preset_results]),
            'avg_step_size': np.mean([r['avg_step'] for r in preset_results]),
        }
        
        results[preset_name] = {
            'per_text': preset_results,
            'summary': avg_stats,
        }
    
    return results


def compare_models(model_paths: list[Path], config: dict, alphabet: str, 
                   device: torch.device, output_dir: Path):
    """여러 모델 비교"""
    all_results = []
    
    for model_path in model_paths:
        print(f"\nEvaluating {model_path.name}...")
        results = evaluate_model(model_path, config, alphabet, device)
        all_results.append(results)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Models\n\n")
        for i, r in enumerate(all_results):
            f.write(f"{i+1}. `{Path(r['model']).name}`\n")
        
        f.write("\n## Summary (preset: normal)\n\n")
        f.write("| Model | Avg Length | Steps/Char | Pen-up Ratio | Aspect Ratio | Step Size |\n")
        f.write("|-------|------------|------------|--------------|--------------|----------|\n")
        
        for r in all_results:
            s = r['normal']['summary']
            name = Path(r['model']).stem[:30]
            f.write(f"| {name} | {s['avg_length']:.1f} | {s['avg_steps_per_char']:.1f} | {s['avg_pen_up_ratio']:.3f} | {s['avg_aspect_ratio']:.2f} | {s['avg_step_size']:.4f} |\n")
        
        for preset in ['normal', 'sharp', 'loose']:
            f.write(f"\n## Detailed: {preset}\n\n")
            
            for r in all_results:
                model_name = Path(r['model']).stem
                f.write(f"### {model_name}\n\n")
                f.write("| Text | Length | Steps/Char | Pen-ups | Aspect |\n")
                f.write("|------|--------|------------|---------|--------|\n")
                
                for tr in r[preset]['per_text']:
                    text_short = tr['text'][:20]
                    f.write(f"| {text_short} | {tr['length']} | {tr['steps_per_char']:.1f} | {tr['pen_up_count']} | {tr['aspect_ratio']:.2f} |\n")
                f.write("\n")
    
    print(f"\nReport saved to {report_path}")
    
    samples_dir = output_dir / 'comparison_samples'
    samples_dir.mkdir(exist_ok=True)
    
    for model_path in model_paths:
        model = load_model(model_path, config, alphabet, device)
        sampler = Sampler(model, alphabet, device)
        model_name = model_path.stem
        
        for i, text in enumerate(EVAL_TEXTS[:3]):
            strokes = sampler.generate_with_preset(text, preset='normal')
            svg = sampler.strokes_to_svg(strokes, width=600, height=120)
            
            safe_text = text.replace(' ', '_')[:15]
            svg_path = samples_dir / f"{model_name}_{i:02d}_{safe_text}.svg"
            svg_path.write_text(svg)
    
    print(f"Sample SVGs saved to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare models')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='Model paths to evaluate')
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--alphabet_file', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if args.alphabet_file:
        if args.alphabet_file.endswith('.h5'):
            import h5py
            with h5py.File(args.alphabet_file, 'r') as f:
                alphabet = f.attrs['alphabet']
        else:
            alphabet = Path(args.alphabet_file).read_text().strip()
    else:
        import h5py
        with h5py.File('data/train.h5', 'r') as f:
            alphabet = f.attrs['alphabet']
    
    model_paths = [Path(m) for m in args.models]
    output_dir = Path(args.output_dir)
    
    compare_models(model_paths, config, alphabet, device, output_dir)


if __name__ == '__main__':
    main()
