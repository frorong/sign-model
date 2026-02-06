import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftWindow(nn.Module):
    def __init__(self, hidden_size: int, num_components: int = 10):
        super().__init__()
        self.num_components = num_components
        self.linear = nn.Linear(hidden_size, 3 * num_components)
    
    def forward(self, h: torch.Tensor, c_seq: torch.Tensor, k_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U = c_seq.size(1)
        
        params = self.linear(h)
        alpha, beta, kappa = params.chunk(3, dim=1)
        
        alpha = F.softplus(alpha) + 1e-4
        beta = F.softplus(beta).clamp(min=0.1, max=10.0)
        kappa = k_prev + F.softplus(kappa) * 0.1
        
        u = torch.arange(U, device=h.device, dtype=torch.float32)[None, None, :]
        a = alpha.unsqueeze(2)
        b = beta.unsqueeze(2)
        k = kappa.unsqueeze(2)
        
        phi = a * torch.exp(-b * (k - u) ** 2)
        phi = phi.sum(dim=1)
        
        window = torch.bmm(phi.unsqueeze(1), c_seq).squeeze(1)
        
        return window, phi, kappa
