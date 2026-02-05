import torch
import torch.nn as nn
import numpy as np


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size: int, num_mixtures: int = 20):
        super().__init__()
        self.num_mixtures = num_mixtures
        
        self.pi_layer = nn.Linear(input_size, num_mixtures)
        self.mu_layer = nn.Linear(input_size, 2 * num_mixtures)
        self.sigma_layer = nn.Linear(input_size, 2 * num_mixtures)
        self.rho_layer = nn.Linear(input_size, num_mixtures)
        self.eos_layer = nn.Linear(input_size, 1)
    
    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        pi = torch.softmax(self.pi_layer(h), dim=-1)
        mu = self.mu_layer(h).view(h.size(0), self.num_mixtures, 2)
        sigma = torch.exp(self.sigma_layer(h)).view(h.size(0), self.num_mixtures, 2)
        rho = torch.tanh(self.rho_layer(h))
        eos = torch.sigmoid(self.eos_layer(h)).squeeze(-1)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'rho': rho,
            'eos': eos
        }
    
    def sample(self, params: dict[str, torch.Tensor], bias: float = 0.0) -> torch.Tensor:
        pi = params['pi']
        mu = params['mu']
        sigma = params['sigma']
        rho = params['rho']
        eos = params['eos']
        
        batch_size = pi.size(0)
        
        if bias > 0:
            pi = torch.softmax(torch.log(pi + 1e-8) * (1 + bias), dim=-1)
            sigma = sigma / (1 + bias)
        
        idx = torch.multinomial(pi, 1).squeeze(-1)
        
        mu_selected = mu[torch.arange(batch_size), idx]
        sigma_selected = sigma[torch.arange(batch_size), idx]
        rho_selected = rho[torch.arange(batch_size), idx]
        
        z1 = torch.randn(batch_size, device=pi.device)
        z2 = torch.randn(batch_size, device=pi.device)
        
        x = mu_selected[:, 0] + sigma_selected[:, 0] * z1
        y = mu_selected[:, 1] + sigma_selected[:, 1] * (rho_selected * z1 + torch.sqrt(1 - rho_selected ** 2) * z2)
        
        eos_sample = (torch.rand(batch_size, device=pi.device) < eos).float()
        
        return torch.stack([x, y, eos_sample], dim=1)
