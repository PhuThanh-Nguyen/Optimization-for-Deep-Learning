import torch
from torch.optim import Optimizer

class SgdOptimizer(Optimizer):
    """Implements SGD Algorithm
    The Nesterov version can be performed by choosing input argument
    """
    def __init__(self, 
        params, 
        lr=1e-3, 
        momentum=0, 
        dampening=0,
        weight_decay=0, 
        nesterov=False):

        defaults = dict(lr=lr, momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay, 
                        nesterov=nesterov)

        super(SgdOptimizer, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss