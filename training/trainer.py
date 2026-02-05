import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from .losses import mdn_loss


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        self.grad_clip_lstm = config['training']['gradient_clip_lstm']
        self.grad_clip_output = config['training']['gradient_clip_output']
    
    def train_epoch(self, dataloader: DataLoader, alphabet: str) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            strokes = batch['strokes'].to(self.device)
            texts = batch['texts']
            stroke_lengths = batch['stroke_lengths']
            
            self.optimizer.zero_grad()
            
            batch_loss = self._process_batch(strokes, texts, stroke_lengths, alphabet)
            batch_loss.backward()
            
            self._clip_gradients()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]
    
    def _process_batch(self, strokes: torch.Tensor, texts: list[str], stroke_lengths: torch.Tensor, alphabet: str) -> torch.Tensor:
        from datasets.preprocessing import text_to_onehot
        
        batch_size = strokes.size(0)
        actual_max_len = int(stroke_lengths.max().item())
        
        c_list = [torch.tensor(text_to_onehot(t, alphabet), device=self.device) for t in texts]
        max_text_len = max(c.size(0) for c in c_list)
        c = torch.zeros(batch_size, max_text_len, len(alphabet), device=self.device)
        for i, c_i in enumerate(c_list):
            c[i, :c_i.size(0)] = c_i
        
        state = None
        total_loss = torch.tensor(0.0, device=self.device)
        total_steps = 0
        
        for t in range(actual_max_len - 1):
            mask = (t + 1 < stroke_lengths).float()
            if mask.sum() == 0:
                break
            
            x = strokes[:, t]
            target = strokes[:, t + 1]
            
            params, state = self.model(x, c, state)
            state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
            
            loss = mdn_loss(target, params)
            total_loss = total_loss + loss * mask.mean()
            total_steps += 1
        
        return total_loss / max(total_steps, 1)
    
    def _clip_gradients(self):
        lstm_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if 'lstm' in name:
                lstm_params.append(param)
            else:
                other_params.append(param)
        
        if lstm_params:
            torch.nn.utils.clip_grad_norm_(lstm_params, self.grad_clip_lstm)
        if other_params:
            torch.nn.utils.clip_grad_norm_(other_params, self.grad_clip_output)
    
    def save_checkpoint(self, path: Path, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
