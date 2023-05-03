import torch
import torch.nn as nn

import src.utils as utils
from src.train import run_episode


class Baseline(nn.Module):
    """The base class for all Baselines."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class RolloutBaseline(Baseline):
    """Computes greedy rollout if 'num_rollouts' is -1 or sampled rollout
    if 'num_rollout' > 0."""
    def __init__(self, num_rollouts=-1, temperature=1, *args, **kwargs):
        super().__init__()
        if (type(num_rollouts) != int) or (num_rollouts < -1) or (num_rollouts == 0):
            raise ValueError(f"{num_rollouts} is not a valid number of rollouts, try with -1 for 'greedy' or 1, 2, 3, etc. for 'sample'.")
        if temperature < 1:
            raise ValueError(f"{temperature} is not a valid number for temperature, try with a flot greater than or equal with 1.")
            

        if num_rollouts == -1:
            self.strategy = 'greedy'
            self.num_rollouts = 1
            self.temperature = 1
        elif num_rollouts >= 1:
            self.strategy = 'sampled'
            self.num_rollouts = num_rollouts
            self.temperature = temperature

    def forward(self, formula, num_variables, policy_network, device, 
                permute_vars, permute_seed, logit_clipping, **kwargs):

        buffer = run_episode(num_variables,
                             policy_network,
                             device,
                             strategy=self.strategy,
                             batch_size=self.num_rollouts,
                             permute_vars=permute_vars,
                             permute_seed=permute_seed,
                             logit_clipping=logit_clipping,
                             logit_temp=self.temperature,
                             extra_logging=False)
        
        mean_num_sat = utils.num_sat_clauses_tensor(formula, buffer.action.detach()).mean().detach()
        return mean_num_sat


class EMABaseline(Baseline):
    """Computes the exponential moving average baseline."""
    def __init__(self, num_clauses, alpha=0.99, *args, **kwargs):
        super().__init__()
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f'`alpha` must be number such that 0 <= `alpha` <= 1, got {alpha}.')
        self.alpha = alpha
        self.b = (7.0/8.0) * num_clauses

    def forward(self, num_sat, **kwargs):
        # num_sat: [batch_size]
        self.b = self.alpha * self.b + (1 - self.alpha) * num_sat.mean()
        return self.b


class ZeroBaseline(Baseline):
    """Returns zero as a baseline"""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, device, **kwargs):
        return torch.tensor(0, dtype=float).detach().to(device)





# class BaselineNet(Baseline):
#     """ """
#     def __init__(self, hidden_size  **kwargs):
#         super().__init__(**kwargs)
#         # Output
#         self.baseline = nn.Linear(hidden_size, 1)

#     def forward(self, formula, num_variables, variables, policy_network, device, dec_state):

#         return self.baseline(dec_state)