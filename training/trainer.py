import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from .losses import mdn_loss, char_prediction_loss
from .optimizers import CustomRMSprop


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device, verbose: bool = False):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.verbose = verbose
        
        self.lr = config['training'].get('learning_rate', 0.0001)
        self.optimizer = CustomRMSprop(
            model.parameters(),
            lr=self.lr,
            alpha=0.95,
            eps=1e-4,
            momentum=0.9,
        )
        
        self.current_epoch = 0
        self._log(f'Trainer initialized: device={device}, optimizer=RMSprop(centered), lr={self.lr}')
    
    def _log(self, msg: str):
        if self.verbose:
            print(f'[Trainer] {msg}')
    
    def train_epoch(self, dataloader: DataLoader, alphabet: str) -> float:
        self.model.train()
        total_loss = 0.0
        self.current_epoch += 1
        self._log(f'Epoch {self.current_epoch} started, batches={len(dataloader)}')
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch}', leave=False)
        for batch in pbar:
            strokes = batch['strokes'].to(self.device)
            texts = batch['texts']
            stroke_lengths = batch['stroke_lengths']
            
            self.optimizer.zero_grad()
            batch_loss = self._process_batch(strokes, texts, stroke_lengths, alphabet)
            batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        self._log(f'Epoch {self.current_epoch} finished, avg_loss={avg_loss:.4f}')
        return avg_loss
    
    def get_lr(self) -> float:
        return self.lr
    
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
        
        for t in range(actual_max_len):
            mask = (t < stroke_lengths).float().to(self.device)
            if mask.sum() == 0:
                break
            
            x = strokes[:, t - 1] if t > 0 else torch.zeros(batch_size, 3, device=self.device)
            target = strokes[:, t]
            
            params, state = self.model(x, c, state)
            mdn = mdn_loss(target, params, mask=mask)
            char_weight = self.config['training'].get('char_loss_weight', 0.1)
            char_loss = char_prediction_loss(state['char_logits'], state['phi'], c, mask=mask)
            total_loss = total_loss + mdn + char_weight * char_loss
        
        return total_loss / batch_size
    
    def _clip_gradients(self):
        lstm_params = list(self.model.lstm1.parameters()) + \
                      list(self.model.lstm2.parameters()) + \
                      list(self.model.lstm3.parameters())
        output_params = list(self.model.mdn.parameters()) + list(self.model.char_head.parameters())
        
        torch.nn.utils.clip_grad_value_(lstm_params, 10)
        torch.nn.utils.clip_grad_value_(output_params, 100)
    
    def save_checkpoint(self, path: Path, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self._log(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self, path: Path, model_only: bool = False) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            if not model_only and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            epoch = 0
        self._log(f'Checkpoint loaded: {path}, epoch={epoch}, model_only={model_only}')
        return epoch
