import torch
from pathlib import Path
from models import SynthesisNetwork


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model: SynthesisNetwork):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        h1: torch.Tensor, c1: torch.Tensor,
        h2: torch.Tensor, c2: torch.Tensor,
        h3: torch.Tensor, c3: torch.Tensor,
        k: torch.Tensor, w: torch.Tensor
    ) -> tuple:
        state = {
            'h1': h1, 'c1': c1,
            'h2': h2, 'c2': c2,
            'h3': h3, 'c3': c3,
            'k': k, 'w': w
        }
        
        params, new_state = self.model(x, c, state)
        
        return (
            params['pi'], params['mu'], params['sigma'], params['rho'], params['eos'],
            new_state['h1'], new_state['c1'],
            new_state['h2'], new_state['c2'],
            new_state['h3'], new_state['c3'],
            new_state['k'], new_state['w']
        )


def export_to_onnx(model: SynthesisNetwork, output_path: Path, alphabet_size: int):
    model.eval()
    wrapper = ONNXWrapper(model)
    
    hidden_size = model.hidden_size
    num_components = model.attention.num_components
    
    x = torch.zeros(1, 3)
    c = torch.zeros(1, 10, alphabet_size)
    h1 = torch.zeros(1, hidden_size)
    c1 = torch.zeros(1, hidden_size)
    h2 = torch.zeros(1, hidden_size)
    c2 = torch.zeros(1, hidden_size)
    h3 = torch.zeros(1, hidden_size)
    c3 = torch.zeros(1, hidden_size)
    k = torch.zeros(1, num_components)
    w = torch.zeros(1, alphabet_size)
    
    torch.onnx.export(
        wrapper,
        (x, c, h1, c1, h2, c2, h3, c3, k, w),
        str(output_path),
        input_names=['x', 'c', 'h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'k', 'w'],
        output_names=['pi', 'mu', 'sigma', 'rho', 'eos', 'h1_out', 'c1_out', 'h2_out', 'c2_out', 'h3_out', 'c3_out', 'k_out', 'w_out'],
        dynamic_axes={
            'c': {1: 'seq_len'},
        },
        opset_version=11
    )
