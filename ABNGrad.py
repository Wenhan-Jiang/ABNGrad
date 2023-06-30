import math
import torch
from torch.optim.optimizer import Optimizer

class ABNGrad(Optimizer):
    r"""Implements ABNGrad algorithm.
    It has been proposed in `ABNGrad:AdaPtive Step Size Gradient Descent for Optimizing Neural Nerworks`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        absgrad (boolean, optional): whether to use the ABNGrad1e-3 variant of this
            algorithm from the paper `......`_
    .. _ABNGrad:AdaPtive Step Size Gradient Descent for Optimizing Neural Nerworks:
        # adam + amsgrad is false
    .. _......:
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, pn=2, abngrad=True, amsgrad=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, pn=pn, abngrad=abngrad, amsgrad=amsgrad)
        super(ABNGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ABNGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('abngrad', False)
            group.setdefault('amsgrad', False)
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                abngrad = group['abngrad']
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq= state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']
                pn = group['pn']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Maintains the abs of all 2nd moment running avg. till now
                if abngrad and amsgrad:
                    # Use the max. for normalizing running avg. of gradient
                    exp_avg_sq.add_((1 - beta2), torch.abs(grad.mul_(grad) - exp_avg_sq))
                    exp_avg_sq.div_(torch.norm(exp_avg_sq, p=pn))
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                elif abngrad:
                    # Use the abs. for normalizing running avg. of gradient
                    exp_avg_sq.add_((1 - beta2), torch.abs(grad.mul_(grad) - exp_avg_sq))
                    exp_avg_sq.div_(torch.norm(exp_avg_sq, p=pn))
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                else: #Adam
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) #Adam
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
