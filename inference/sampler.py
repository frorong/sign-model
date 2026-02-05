import torch
import numpy as np
from models import SynthesisNetwork
from datasets.preprocessing import text_to_onehot


class Sampler:
    def __init__(self, model: SynthesisNetwork, alphabet: str, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.alphabet = alphabet
        self.device = device
    
    @torch.no_grad()
    def generate(self, text: str, max_steps: int = 700, bias: float = 0.0, min_steps: int = 50, progress_callback=None) -> np.ndarray:
        c = torch.tensor(text_to_onehot(text, self.alphabet), device=self.device)
        c = c.unsqueeze(0)
        
        x = torch.zeros(1, 3, device=self.device)
        state = None
        
        strokes = []
        
        for step in range(max_steps):
            params, state = self.model(x, c, state)
            
            x = self.model.mdn.sample(params, bias=bias)
            strokes.append(x.cpu().numpy()[0])
            
            if progress_callback:
                progress_callback(step + 1, max_steps)
            
            if step >= min_steps:
                phi = state['phi']
                if phi[0, -1] > 0.95:
                    break
                
                if x[0, 2] > 0.5:
                    break
        
        return np.array(strokes)
    
    def strokes_to_coords(self, strokes: np.ndarray) -> np.ndarray:
        coords = np.cumsum(strokes[:, :2], axis=0)
        return coords
    
    def strokes_to_svg_path(self, strokes: np.ndarray, scale: float = 1.0) -> str:
        coords = self.strokes_to_coords(strokes)
        coords = coords * scale
        
        path_parts = [f"M {coords[0, 0]:.2f} {coords[0, 1]:.2f}"]
        
        pen_up = False
        for i in range(1, len(coords)):
            if pen_up:
                path_parts.append(f"M {coords[i, 0]:.2f} {coords[i, 1]:.2f}")
                pen_up = False
            else:
                path_parts.append(f"L {coords[i, 0]:.2f} {coords[i, 1]:.2f}")
            
            if strokes[i, 2] > 0.5:
                pen_up = True
        
        return " ".join(path_parts)
