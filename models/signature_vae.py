from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignatureEncoder(nn.Module):
    """
    서명 스트로크를 잠재 공간으로 인코딩
    스타일 특징 추출
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 256, latent_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc_mu = nn.Linear(hidden_size * 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_size)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.lstm(x_packed)
        else:
            _, (h, _) = self.lstm(x)
        
        h = torch.cat([h[-2], h[-1]], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class SignatureDecoder(nn.Module):
    """
    잠재 벡터에서 서명 스트로크 생성
    """
    def __init__(self, latent_size: int = 64, hidden_size: int = 256, output_size: int = 3, num_layers: int = 2):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.fc_init = nn.Linear(latent_size, hidden_size * num_layers * 2)
        
        self.lstm = nn.LSTM(
            input_size=output_size + latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, z: torch.Tensor, target: torch.Tensor = None, max_len: int = 500) -> torch.Tensor:
        batch_size = z.size(0)
        
        init_state = self.fc_init(z)
        h = init_state[:, :self.hidden_size * self.num_layers].view(self.num_layers, batch_size, self.hidden_size).contiguous()
        c = init_state[:, self.hidden_size * self.num_layers:].view(self.num_layers, batch_size, self.hidden_size).contiguous()
        
        if target is not None:
            seq_len = target.size(1)
            z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([target, z_expanded], dim=2)
            
            output, _ = self.lstm(lstm_input, (h, c))
            output = self.fc_out(output)
            return output
        
        outputs = []
        x = torch.zeros(batch_size, 1, 3, device=z.device)
        
        for _ in range(max_len):
            z_step = z.unsqueeze(1)
            lstm_input = torch.cat([x, z_step], dim=2)
            
            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = self.fc_out(out)
            outputs.append(out)
            
            x = out.detach()
        
        return torch.cat(outputs, dim=1)


class SignatureVAE(nn.Module):
    """
    서명 스타일을 학습하는 VAE
    
    Stage 2에서 사용:
    - 인코더: 서명 → 스타일 잠재 벡터
    - 디코더: 스타일 벡터 + 입력 스트로크 → 스타일화된 스트로크
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 256, latent_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = SignatureEncoder(input_size, hidden_size, latent_size, num_layers)
        self.decoder = SignatureDecoder(latent_size, hidden_size, input_size, num_layers)
        self.latent_size = latent_size
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        
        x_shifted = torch.zeros_like(x)
        x_shifted[:, 1:] = x[:, :-1]
        
        recon = self.decoder(z, x_shifted)
        
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        mu, logvar = self.encoder(x, lengths)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor, max_len: int = 500) -> torch.Tensor:
        return self.decoder(z, max_len=max_len)
    
    def sample(self, num_samples: int = 1, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_size, device=device)
        return self.decode(z)


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, 
             mask: torch.Tensor = None, kl_weight: float = 0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 손실 함수
    """
    recon_loss = F.mse_loss(recon[:, :, :2], target[:, :, :2], reduction='none')
    eos_loss = F.binary_cross_entropy_with_logits(recon[:, :, 2], target[:, :, 2], reduction='none')
    
    total_recon = recon_loss.sum(dim=-1) + eos_loss
    
    if mask is not None:
        total_recon = (total_recon * mask).sum() / mask.sum().clamp(min=1)
    else:
        total_recon = total_recon.mean()
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = total_recon + kl_weight * kl_loss
    
    return total_loss, total_recon, kl_loss
