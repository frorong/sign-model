import torch
import torch.nn as nn


class SoftWindow(nn.Module):
    def __init__(self, hidden_size: int, num_components: int = 10):
        super().__init__()
        self.num_components = num_components
        self.linear = nn.Linear(hidden_size, 3 * num_components)
    
    def forward(self, h: torch.Tensor, c_seq: torch.Tensor, k_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = h.size(0)
        seq_len = c_seq.size(1)
        
        params = self.linear(h)
        alpha, beta, kappa = params.chunk(3, dim=1)
        
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        kappa = k_prev + torch.exp(kappa)
        
        u = torch.arange(seq_len, device=h.device, dtype=torch.float32)
        u = u.unsqueeze(0).unsqueeze(0)
        
        kappa_expanded = kappa.unsqueeze(2)
        alpha_expanded = alpha.unsqueeze(2)
        beta_expanded = beta.unsqueeze(2)
        
        phi = alpha_expanded * torch.exp(-beta_expanded * (kappa_expanded - u) ** 2)
        phi = phi.sum(dim=1)
        
        window = torch.bmm(phi.unsqueeze(1), c_seq).squeeze(1)
        
        return window, phi, kappa
