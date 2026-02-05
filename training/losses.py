import torch
import numpy as np


def mdn_loss(target: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    x_target = target[:, 0:1]
    y_target = target[:, 1:2]
    eos_target = target[:, 2]
    
    pi = params['pi']
    mu = params['mu']
    sigma = params['sigma']
    rho = params['rho']
    eos = params['eos']
    
    mu_x = mu[:, :, 0]
    mu_y = mu[:, :, 1]
    sigma_x = sigma[:, :, 0]
    sigma_y = sigma[:, :, 1]
    
    z_x = (x_target - mu_x) / sigma_x
    z_y = (y_target - mu_y) / sigma_y
    z = z_x ** 2 + z_y ** 2 - 2 * rho * z_x * z_y
    
    norm_factor = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho ** 2)
    exp_term = torch.exp(-z / (2 * (1 - rho ** 2)))
    
    gaussian = exp_term / norm_factor
    mixture = torch.sum(pi * gaussian, dim=1)
    
    stroke_loss = -torch.log(mixture + 1e-8)
    
    eos_loss = -eos_target * torch.log(eos + 1e-8) - (1 - eos_target) * torch.log(1 - eos + 1e-8)
    
    return (stroke_loss + eos_loss).mean()
