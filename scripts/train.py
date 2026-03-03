#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from datetime import datetime

import time
from datasets import IAMOnDBDataset, collate_fn
from models import SynthesisNetwork
from training import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train handwriting synthesis model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training HDF5 data file')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation HDF5 data file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.01, help='Minimum delta for early stopping')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model_only', action='store_true', help='Load only model weights from checkpoint (skip optimizer state)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    num_workers = config['training'].get('num_workers', 4)
    
    train_dataset = IAMOnDBDataset(args.train_data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    )
    
    val_dataloader = None
    if args.val_data:
        val_dataset = IAMOnDBDataset(args.val_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn
        )
    
    alphabet = train_dataset.alphabet
    print(f'Alphabet size: {len(alphabet)}')
    print(f'Alphabet: {repr(alphabet[:50])}...')
    
    model = SynthesisNetwork(
        alphabet_size=len(alphabet),
        hidden_size=config['model']['hidden_size'],
        num_mixtures=config['model']['num_mixtures'],
        num_attention_components=config['model']['num_attention_components']
    )
    
    trainer = Trainer(model, config, device, verbose=args.verbose)
    
    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(Path(args.checkpoint), model_only=args.model_only)
        print(f'Resumed from epoch {start_epoch} (model_only={args.model_only})')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_model_path = output_dir / 'best_model.pt'
    final_path = output_dir / f'model_final_{timestamp}.pt'
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        train_loss = trainer.train_epoch(train_dataloader, alphabet)
        epoch_time = time.time() - epoch_start
        lr = trainer.get_lr()
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}, Train Loss: {train_loss:.4f}, LR: {lr:.6f}, Time: {epoch_time:.1f}s', end='')
        
        if val_dataloader:
            trainer.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    strokes = batch['strokes'].to(device)
                    texts = batch['texts']
                    stroke_lengths = batch['stroke_lengths']
                    batch_loss = trainer._process_batch(strokes, texts, stroke_lengths, alphabet)
                    val_loss += batch_loss.item()
            val_loss /= len(val_dataloader)
            print(f', Val Loss: {val_loss:.4f}')
            current_loss = val_loss
        else:
            print()
            current_loss = train_loss
        
        if current_loss < best_loss - args.early_stop_min_delta:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            loss_type = 'val' if val_dataloader else 'train'
            print(f'  → New best model saved ({loss_type}_loss: {current_loss:.4f})')
        else:
            patience_counter += 1
            if val_dataloader and patience_counter >= args.early_stop_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs (patience: {args.early_stop_patience})')
                print(f'Best validation loss: {best_loss:.4f}')
                break
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    torch.save(model.state_dict(), final_path)
    print(f'\nSaved final model: {final_path}')
    if best_model_path.exists():
        loss_type = 'val' if val_dataloader else 'train'
        print(f'Best model (lowest {loss_type} loss): {best_model_path}')


if __name__ == '__main__':
    main()
