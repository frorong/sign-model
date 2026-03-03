import torch
import torch.nn as nn
from .lstm import PeepholeLSTM
from .attention import SoftWindow
from .mdn import MixtureDensityLayer


class SynthesisNetwork(nn.Module):
    def __init__(self, alphabet_size: int, hidden_size: int = 400, num_mixtures: int = 20, num_attention_components: int = 10, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        
        self.lstm1 = PeepholeLSTM(3 + alphabet_size, hidden_size)
        self.attention = SoftWindow(hidden_size, num_attention_components)
        self.lstm2 = PeepholeLSTM(3 + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(3 + hidden_size + alphabet_size, hidden_size)
        self.mdn = MixtureDensityLayer(hidden_size * 3, num_mixtures)
        self.char_head = nn.Linear(hidden_size * 3, alphabet_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, state: dict = None) -> tuple[dict, dict]:
        batch_size = x.size(0)
        
        if state is None:
            state = self.init_state(batch_size, x.device)
        
        h1, c1 = state['h1'], state['c1']
        h2, c2 = state['h2'], state['c2']
        h3, c3 = state['h3'], state['c3']
        k = state['k']
        w = state['w']
        
        lstm1_input = torch.cat([x, w.squeeze(1)], dim=1)
        h1, (h1, c1) = self.lstm1(lstm1_input, (h1, c1))
        
        w, phi, k = self.attention(h1, c, k)
        
        lstm2_input = torch.cat([x, self.dropout1(h1), w.squeeze(1)], dim=1)
        h2, (h2, c2) = self.lstm2(lstm2_input, (h2, c2))
        
        lstm3_input = torch.cat([x, self.dropout2(h2), w.squeeze(1)], dim=1)
        h3, (h3, c3) = self.lstm3(lstm3_input, (h3, c3))
        
        h_combined = torch.cat([h1, h2, h3], dim=1)
        mdn_params = self.mdn(h_combined)
        char_logits = self.char_head(h_combined)
        
        new_state = {
            'h1': h1, 'c1': c1,
            'h2': h2, 'c2': c2,
            'h3': h3, 'c3': c3,
            'k': k, 'w': w, 'phi': phi, 'char_logits': char_logits
        }
        
        return mdn_params, new_state
    
    def init_state(self, batch_size: int, device: torch.device) -> dict:
        zeros_h = lambda: torch.zeros(batch_size, self.hidden_size, device=device)
        return {
            'h1': zeros_h(), 'c1': zeros_h(),
            'h2': zeros_h(), 'c2': zeros_h(),
            'h3': zeros_h(), 'c3': zeros_h(),
            'k': torch.zeros(batch_size, self.attention.num_components, device=device),
            'w': torch.zeros(batch_size, 1, self.alphabet_size, device=device)
        }
