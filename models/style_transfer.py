from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleConditionedTransfer(nn.Module):
    """
    스타일 조건부 스트로크 변환
    
    입력: 
        - handwriting strokes (Stage 1 출력)
        - style vector (SignatureVAE에서 추출)
    출력:
        - stylized strokes (서명 스타일)
    """
    def __init__(self, stroke_size: int = 3, style_size: int = 64, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        
        self.style_mlp = nn.Sequential(
            nn.Linear(style_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.encoder = nn.LSTM(
            input_size=stroke_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.style_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, stroke_size)
        
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size + stroke_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, strokes: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = strokes.size()
        
        style_features = self.style_mlp(style)
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                strokes, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoded, _ = self.encoder(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=seq_len)
        else:
            encoded, _ = self.encoder(strokes)
        
        style_query = style_features.unsqueeze(1).expand(-1, seq_len, -1)
        style_query = F.pad(style_query, (0, encoded.size(-1) - style_query.size(-1)))
        
        attended, _ = self.style_attention(
            query=encoded,
            key=encoded,
            value=encoded
        )
        
        style_expanded = style_features.unsqueeze(1).expand(-1, seq_len, -1)
        decoder_input = torch.cat([attended, style_expanded], dim=-1)
        
        decoded, _ = self.decoder(decoder_input)
        
        output = self.output_layer(decoded)
        
        gate_input = torch.cat([decoded, strokes], dim=-1)
        gate = self.residual_gate(gate_input)
        
        final_output = gate * output + (1 - gate) * strokes
        
        return final_output


class SignatureTransferPipeline(nn.Module):
    """
    전체 서명 생성 파이프라인
    
    Stage 1 (SynthesisNetwork) + Stage 2 (StyleTransfer) 통합
    """
    def __init__(self, synthesis_model: nn.Module, style_vae: nn.Module, transfer_model: nn.Module):
        super().__init__()
        self.synthesis = synthesis_model
        self.style_vae = style_vae
        self.transfer = transfer_model
    
    def forward(self, text_onehot: torch.Tensor, style_strokes: torch.Tensor = None, 
                style_vector: torch.Tensor = None) -> torch.Tensor:
        """
        text_onehot: 텍스트 원핫 인코딩
        style_strokes: 스타일 참조 서명 (VAE로 인코딩됨)
        style_vector: 직접 제공하는 스타일 벡터
        """
        raise NotImplementedError("Use generate() for inference")
    
    @torch.no_grad()
    def generate(self, text: str, alphabet: str, 
                 style_strokes: torch.Tensor = None,
                 style_vector: torch.Tensor = None,
                 max_steps: int = 500,
                 bias: float = 1.0) -> torch.Tensor:
        """
        텍스트에서 서명 스타일 스트로크 생성
        """
        from datasets.preprocessing import text_to_onehot
        
        device = next(self.synthesis.parameters()).device
        
        c = torch.tensor(text_to_onehot(text, alphabet), device=device).unsqueeze(0)
        
        x = torch.zeros(1, 3, device=device)
        state = None
        base_strokes = []
        
        for step in range(max_steps):
            params, state = self.synthesis(x, c, state)
            x = self.synthesis.mdn.sample(params, bias=bias)
            base_strokes.append(x)
            
            kappa_mean = state['k'].mean().item()
            if kappa_mean > len(text) * 1.2 and step > 50:
                break
        
        base_strokes = torch.stack(base_strokes, dim=1)
        
        if style_vector is None:
            if style_strokes is not None:
                style_vector = self.style_vae.encode(style_strokes)
            else:
                style_vector = torch.randn(1, self.style_vae.latent_size, device=device)
        
        styled_strokes = self.transfer(base_strokes, style_vector)
        
        return styled_strokes


def transfer_loss(output: torch.Tensor, target: torch.Tensor, 
                  base_strokes: torch.Tensor,
                  mask: torch.Tensor = None,
                  content_weight: float = 1.0,
                  smooth_weight: float = 0.1) -> torch.Tensor:
    """
    스타일 전이 손실 함수
    
    - 재구성 손실: 타겟 서명과의 유사도
    - 스무딩 손실: 출력 스트로크의 부드러움
    """
    recon_loss = F.mse_loss(output[:, :, :2], target[:, :, :2], reduction='none')
    eos_loss = F.binary_cross_entropy_with_logits(output[:, :, 2], target[:, :, 2], reduction='none')
    
    total_recon = recon_loss.sum(dim=-1) + eos_loss
    
    if mask is not None:
        total_recon = (total_recon * mask).sum() / mask.sum().clamp(min=1)
    else:
        total_recon = total_recon.mean()
    
    velocity = output[:, 1:, :2] - output[:, :-1, :2]
    accel = velocity[:, 1:] - velocity[:, :-1]
    smooth_loss = (accel ** 2).mean()
    
    total_loss = content_weight * total_recon + smooth_weight * smooth_loss
    
    return total_loss
