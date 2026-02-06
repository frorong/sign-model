#!/usr/bin/env python3
"""에포크별 체크포인트에서 샘플 생성 및 저장"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import yaml
from models import SynthesisNetwork
from inference.sampler import Sampler, PRESETS


TEST_TEXTS = [
    "hello world",
    "The quick brown fox",
    "Signature Model",
    "ABCDEFG",
    "123456789",
]


def load_model(checkpoint_path: Path, config: dict, alphabet: str, device: torch.device) -> SynthesisNetwork:
    model = SynthesisNetwork(
        alphabet_size=len(alphabet),
        hidden_size=config['model']['hidden_size'],
        num_mixtures=config['model']['num_mixtures'],
        num_attention_components=config['model']['num_attention_components']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def sample_from_checkpoint(checkpoint_path: Path, config: dict, alphabet: str, 
                           output_dir: Path, device: torch.device, presets: list[str] = None):
    if presets is None:
        presets = ['normal', 'sharp']
    
    model = load_model(checkpoint_path, config, alphabet, device)
    sampler = Sampler(model, alphabet, device)
    
    checkpoint_name = checkpoint_path.stem
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for preset in presets:
        for i, text in enumerate(TEST_TEXTS):
            strokes = sampler.generate_with_preset(text, preset=preset)
            svg = sampler.strokes_to_svg(strokes, width=600, height=120)
            
            safe_text = text.replace(' ', '_')[:20]
            svg_path = checkpoint_dir / f"{i:02d}_{safe_text}_{preset}.svg"
            svg_path.write_text(svg)
    
    print(f"Saved samples to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from checkpoints')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='samples')
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to sample')
    parser.add_argument('--alphabet_file', type=str, default=None, help='Path to alphabet file or H5 data')
    parser.add_argument('--presets', type=str, nargs='+', default=['normal', 'sharp'])
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
        h5_path = Path('data/train.h5')
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                alphabet = f.attrs['alphabet']
        else:
            alphabet = " !\"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        sample_from_checkpoint(checkpoint_path, config, alphabet, output_dir, device, args.presets)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        
        print(f"Found {len(checkpoints)} checkpoints")
        for cp in checkpoints:
            print(f"\nProcessing {cp.name}...")
            sample_from_checkpoint(cp, config, alphabet, output_dir, device, args.presets)


if __name__ == '__main__':
    main()
