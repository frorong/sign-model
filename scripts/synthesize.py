#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import torch
import numpy as np
import h5py

from models import SynthesisNetwork
from inference import Sampler


DEFAULT_ALPHABET = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'


def main():
    parser = argparse.ArgumentParser(description='Synthesize handwriting')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.svg', help='Output SVG file')
    parser.add_argument('--bias', type=float, default=0.5, help='Sampling bias (0=random, higher=more deterministic)')
    parser.add_argument('--scale', type=float, default=1.0, help='Output scale')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SynthesisNetwork(
        alphabet_size=config['model']['alphabet_size'],
        hidden_size=config['model']['hidden_size'],
        num_mixtures=config['model']['num_mixtures'],
        num_attention_components=config['model']['num_attention_components']
    )
    
    state_dict = torch.load(args.model, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    
    mean, std = None, None
    train_path = project_root / "data" / "train_words.h5"
    if not train_path.exists():
        train_path = project_root / "data" / "train.h5"
    if train_path.exists():
        with h5py.File(train_path, 'r') as f:
            if 'mean' in f and 'std' in f:
                mean = f['mean'][:]
                std = f['std'][:]
    
    sampler = Sampler(model, DEFAULT_ALPHABET, device, mean=mean, std=std)
    
    print(f'Generating "{args.text}"...')
    strokes = sampler.generate(args.text, bias=args.bias)
    
    svg_path = sampler.strokes_to_svg_path(strokes, scale=args.scale)
    
    coords = sampler.strokes_to_coords(strokes) * args.scale
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    width = max_x - min_x + 20
    height = max_y - min_y + 20
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="{min_x - 10:.0f} {min_y - 10:.0f} {width:.0f} {height:.0f}">
  <path d="{svg_path}" fill="none" stroke="black" stroke-width="2"/>
</svg>'''
    
    output_path = Path(args.output)
    output_path.write_text(svg_content)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
