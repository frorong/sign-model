import torch
import torch.nn.functional as F
import math


def mdn_loss(target: torch.Tensor, params: dict[str, torch.Tensor], mask: torch.Tensor = None) -> torch.Tensor:
    eps = 1e-6
    
    x_t = target[:, 0:1]
    y_t = target[:, 1:2]
    eos_t = target[:, 2].clamp(0.0, 1.0)
    
    pi = params['pi'].clamp_min(eps)
    mu = params['mu']
    sigma = params['sigma'].clamp_min(1e-3)
    rho = params['rho'].clamp(-0.999, 0.999)
    eos = params['eos'].clamp(eps, 1.0 - eps)
    
    mu_x = mu[:, :, 0]
    mu_y = mu[:, :, 1]
    sx = sigma[:, :, 0]
    sy = sigma[:, :, 1]
    
    zx = (x_t - mu_x) / sx
    zy = (y_t - mu_y) / sy
    
    one_minus_rho2 = (1.0 - rho * rho).clamp_min(1e-5)
    z = zx * zx + zy * zy - 2.0 * rho * zx * zy
    
    log_norm = -torch.log(2.0 * math.pi * sx * sy * torch.sqrt(one_minus_rho2))
    log_exp = -z / (2.0 * one_minus_rho2)
    log_gauss = log_norm + log_exp
    
    log_mix = torch.log(pi) + log_gauss
    stroke_nll = -torch.logsumexp(log_mix, dim=1)
    
    eos_bce = F.binary_cross_entropy(eos, eos_t, reduction='none')
    
    loss = stroke_nll + 2.0 * eos_bce
    
    if mask is not None:
        mask = mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    
    return loss.mean()
