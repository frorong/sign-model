#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np

from models.signature_vae import SignatureVAE, vae_loss


def load_signature_data(h5_path: str, device: torch.device):
    with h5py.File(h5_path, 'r') as f:
        strokes = torch.tensor(f['strokes'][:], dtype=torch.float32)
        lengths = torch.tensor(f['lengths'][:], dtype=torch.long)
        user_indices = torch.tensor(f['user_indices'][:], dtype=torch.long)
        num_users = f.attrs['num_users']
    
    return strokes, lengths, user_indices, num_users


def create_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < lengths.unsqueeze(1)).float()
    return mask


def train_vae(model: SignatureVAE, train_loader: DataLoader, val_loader: DataLoader,
              config: dict, device: torch.device, checkpoint_dir: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for batch in train_loader:
            strokes, lengths = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(strokes, lengths)
            
            mask = create_mask(lengths, strokes.size(1), device)
            loss, recon_loss, kl_loss = vae_loss(
                recon, strokes, mu, logvar, mask,
                kl_weight=config['training']['kl_weight']
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                strokes, lengths = batch[0].to(device), batch[1].to(device)
                recon, mu, logvar = model(strokes, lengths)
                mask = create_mask(lengths, strokes.size(1), device)
                loss, _, _ = vae_loss(recon, strokes, mu, logvar, mask, kl_weight=config['training']['kl_weight'])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, "
              f"Train: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}), "
              f"Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_dir / 'best_signature_vae.pt')
            print(f"  → New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir / f'signature_vae_epoch_{epoch+1}.pt')
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/signature.yaml')
    parser.add_argument('--data', type=str, required=True, help='Path to signature H5 file')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    strokes, lengths, user_indices, num_users = load_signature_data(args.data, device)
    print(f"Loaded {len(strokes)} signatures from {num_users} users")
    
    n_train = int(len(strokes) * config['data']['train_split'])
    indices = torch.randperm(len(strokes))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = TensorDataset(strokes[train_indices], lengths[train_indices])
    val_dataset = TensorDataset(strokes[val_indices], lengths[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    model = SignatureVAE(
        input_size=3,
        hidden_size=config['model']['hidden_size'],
        latent_size=config['model']['latent_size'],
        num_layers=config['model']['num_layers']
    ).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    train_vae(model, train_loader, val_loader, config, device, checkpoint_dir)


if __name__ == '__main__':
    main()
