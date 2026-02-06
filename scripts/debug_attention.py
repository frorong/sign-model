#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt

from models import SynthesisNetwork
from datasets.preprocessing import text_to_onehot


def main():
    train_path = project_root / "data" / "train.h5"
    with h5py.File(train_path, 'r') as f:
        alphabet = f.attrs['alphabet']
    
    config_path = project_root / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = SynthesisNetwork(
        alphabet_size=len(alphabet),
        hidden_size=config['model']['hidden_size'],
        num_mixtures=config['model']['num_mixtures'],
        num_attention_components=config['model']['num_attention_components']
    )
    
    checkpoint_dir = project_root / "checkpoints"
    models = sorted(checkpoint_dir.glob("best_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not models:
        print("No checkpoint found")
        return
    
    model_path = models[0]
    print(f"Loading: {model_path.name}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    text = "Seungje Lee"
    print(f"Text: '{text}' ({len(text)} chars)")
    
    c = torch.tensor(text_to_onehot(text, alphabet), device=device).unsqueeze(0)
    print(f"One-hot shape: {c.shape}")
    
    x = torch.zeros(1, 3, device=device)
    state = None
    
    all_phi = []
    all_kappa = []
    strokes = []
    
    max_steps = 300
    
    min_steps = 50
    
    with torch.no_grad():
        for step in range(max_steps):
            params, state = model(x, c, state)
            
            phi = state['phi'].cpu().numpy()[0]
            kappa = state['k'].cpu().numpy()[0]
            kappa_mean = kappa.mean()
            
            all_phi.append(phi)
            all_kappa.append(kappa_mean)
            
            x = model.mdn.sample(params, bias=0.5)
            strokes.append(x.cpu().numpy()[0])
            
            if step >= min_steps:
                if kappa_mean > len(text) and state['phi'][0, -1] > 0.8:
                    print(f"Stopped at step {step}: kappa ({kappa_mean:.2f}) > text_len and phi[-1] > 0.8")
                    break
                if x[0, 2] > 0.8 and kappa_mean > len(text) * 0.9:
                    print(f"Stopped at step {step}: eos > 0.8 and kappa > 90% text_len")
                    break
    
    all_phi = np.array(all_phi)
    all_kappa = np.array(all_kappa)
    strokes = np.array(strokes)
    
    print(f"\nGenerated {len(strokes)} steps")
    print(f"Phi shape: {all_phi.shape}")
    print(f"Kappa range: {all_kappa.min():.2f} ~ {all_kappa.max():.2f}")
    print(f"Kappa final: {all_kappa[-1]:.2f} (should be ~{len(text)} if reading full text)")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    ax1 = axes[0]
    im = ax1.imshow(all_phi.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Character position')
    ax1.set_title('Attention (phi) over time')
    ax1.set_yticks(range(len(text)))
    ax1.set_yticklabels(list(text))
    plt.colorbar(im, ax=ax1)
    
    ax2 = axes[1]
    ax2.plot(all_kappa, label='kappa (mean)')
    ax2.axhline(y=len(text), color='r', linestyle='--', label=f'Text length ({len(text)})')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Kappa (attention position)')
    ax2.set_title('Attention position over time')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = axes[2]
    coords = np.cumsum(strokes[:, :2], axis=0)
    pen_ups = np.where(strokes[:, 2] > 0.5)[0]
    
    start = 0
    for end in list(pen_ups) + [len(coords)]:
        if end > start:
            ax3.plot(coords[start:end+1, 0], -coords[start:end+1, 1], 'b-', linewidth=1.5)
        start = end + 1
    
    ax3.set_aspect('equal')
    ax3.set_title(f'Generated strokes for "{text}"')
    ax3.axis('off')
    
    plt.tight_layout()
    
    output_path = project_root / "debug_attention.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
