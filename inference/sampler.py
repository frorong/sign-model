import torch
import numpy as np
from models import SynthesisNetwork
from datasets.preprocessing import text_to_onehot


PRESETS = {
    'sharp': {'bias': 0.8, 'description': '선명하고 일관된 출력'},
    'normal': {'bias': 0.5, 'description': '균형잡힌 기본값'},
    'loose': {'bias': 0.3, 'description': '자연스럽고 다양한 변화'},
    'wild': {'bias': 0.1, 'description': '예측 불가능한 스타일'},
}


class Sampler:
    def __init__(self, model: SynthesisNetwork, alphabet: str, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.alphabet = alphabet
        self.device = device
    
    @torch.no_grad()
    def generate(self, text: str, max_steps: int = 700, bias: float = 1.0, 
                 min_steps: int = 50, progress_callback=None) -> np.ndarray:
        c = torch.tensor(text_to_onehot(text, self.alphabet), device=self.device)
        c = c.unsqueeze(0)
        text_len = len(text)
        
        x = torch.zeros(1, 3, device=self.device)
        state = None
        
        strokes = []
        done_streak = 0
        
        for step in range(max_steps):
            params, state = self.model(x, c, state)
            
            x = self.model.mdn.sample(params, bias=bias)
            strokes.append(x.cpu().numpy()[0])
            
            if progress_callback:
                progress_callback(step + 1, max_steps)
            
            if step >= min_steps:
                phi = state['phi']
                kappa_mean = state['k'].mean().item()
                
                phi_at_end = phi[0, -1].item() if phi.size(1) > 0 else 0
                
                if kappa_mean > (text_len - 1) and phi_at_end > 0.6:
                    done_streak += 1
                else:
                    done_streak = 0
                
                if done_streak >= 15 and x[0, 2].item() > 0.7:
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
    
    def generate_with_preset(self, text: str, preset: str = 'normal', **kwargs) -> np.ndarray:
        if preset not in PRESETS:
            preset = 'normal'
        bias = PRESETS[preset]['bias']
        return self.generate(text, bias=bias, **kwargs)
    
    def normalize_strokes(self, strokes: np.ndarray) -> np.ndarray:
        coords = self.strokes_to_coords(strokes)
        
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords = np.where(range_coords < 1e-6, 1.0, range_coords)
        
        normalized = (coords - min_coords) / range_coords.max()
        
        deltas = np.zeros_like(normalized)
        deltas[0] = normalized[0]
        deltas[1:] = normalized[1:] - normalized[:-1]
        
        result = np.zeros_like(strokes)
        result[:, :2] = deltas
        result[:, 2] = strokes[:, 2]
        
        return result
    
    def strokes_to_svg(self, strokes: np.ndarray, width: int = 400, height: int = 100, 
                       padding: int = 10, stroke_width: float = 2.0, color: str = 'black') -> str:
        coords = self.strokes_to_coords(strokes)
        
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        if range_x < 1e-6:
            range_x = 1.0
        if range_y < 1e-6:
            range_y = 1.0
        
        canvas_w = width - 2 * padding
        canvas_h = height - 2 * padding
        
        scale = min(canvas_w / range_x, canvas_h / range_y)
        
        scaled = (coords - [min_x, min_y]) * scale
        offset_x = padding + (canvas_w - range_x * scale) / 2
        offset_y = padding + (canvas_h - range_y * scale) / 2
        scaled += [offset_x, offset_y]
        
        path_parts = [f"M {scaled[0, 0]:.2f} {scaled[0, 1]:.2f}"]
        pen_up = False
        
        for i in range(1, len(scaled)):
            x, y = scaled[i]
            if pen_up:
                path_parts.append(f"M {x:.2f} {y:.2f}")
                pen_up = False
            else:
                path_parts.append(f"L {x:.2f} {y:.2f}")
            if strokes[i, 2] > 0.5:
                pen_up = True
        
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <path d="{' '.join(path_parts)}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    
    def batch_generate(self, texts: list[str], preset: str = 'normal') -> list[np.ndarray]:
        return [self.generate_with_preset(text, preset) for text in texts]
