import torch
import torch.nn as nn


class SoftWindow(nn.Module):
    def __init__(self, hidden_size: int, num_components: int = 10):
        super().__init__()
        self.num_components = num_components
        self.alpha_layer = nn.Linear(hidden_size, num_components)
        self.beta_layer = nn.Linear(hidden_size, num_components)
        self.kappa_layer = nn.Linear(hidden_size, num_components)
    
    def forward(self, h: torch.Tensor, c_seq: torch.Tensor, k_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U = c_seq.size(1)
        
        alpha = torch.exp(self.alpha_layer(h))
        beta = torch.exp(self.beta_layer(h))
        kappa = k_prev + torch.exp(self.kappa_layer(h))
        
        u = torch.arange(U, device=h.device, dtype=torch.float32)[None, None, :]
        a = alpha.unsqueeze(2)
        b = beta.unsqueeze(2)
        k = kappa.unsqueeze(2)
        
        phi = a * torch.exp(-b * (k - u) ** 2)
        phi = phi.sum(dim=1, keepdim=True)
        
        window = torch.bmm(phi, c_seq)
        
        return window, phi, kappa
