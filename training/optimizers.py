import torch
from torch import Tensor


def _rmsprop_step(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffers: list[Tensor],
    lr: float,
    alpha: float,
    eps: float,
    momentum: float,
):
    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        grad_avg = grad_avgs[i]
        grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
        avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add_(eps).sqrt_()

        if momentum > 0:
            buf = momentum_buffers[i]
            buf.mul_(0.9).addcdiv_(grad, avg, value=-lr)
            param.add_(buf)
        else:
            param.addcdiv_(grad, avg, value=-lr)


class CustomRMSprop(torch.optim.Optimizer):
    """RMSprop with epsilon inside sqrt, matching the reference toolkit."""

    def __init__(self, params, lr=1e-4, alpha=0.95, eps=1e-4, momentum=0.9):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffers = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p)
                    state['grad_avg'] = torch.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                square_avgs.append(state['square_avg'])
                grad_avgs.append(state['grad_avg'])
                if group['momentum'] > 0:
                    momentum_buffers.append(state['momentum_buffer'])

                state['step'] += 1

            _rmsprop_step(
                params_with_grad, grads, square_avgs, grad_avgs, momentum_buffers,
                group['lr'], group['alpha'], group['eps'], group['momentum'],
            )

        return loss
