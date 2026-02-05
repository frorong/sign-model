import torch
import torch.nn as nn


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.p_i = nn.Parameter(torch.zeros(hidden_size))
        self.p_f = nn.Parameter(torch.zeros(hidden_size))
        self.p_o = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = state
        
        combined = torch.cat([x, h], dim=1)
        
        i = torch.sigmoid(self.W_i(combined) + self.p_i * c)
        f = torch.sigmoid(self.W_f(combined) + self.p_f * c)
        c_candidate = torch.tanh(self.W_c(combined))
        c_new = f * c + i * c_candidate
        o = torch.sigmoid(self.W_o(combined) + self.p_o * c_new)
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
