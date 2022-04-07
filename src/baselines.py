import torch
import torch.nn as nn

import src.utils as utils


class Baseline(nn.Module):
    """The base class for all Baselines."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BaselineRollout(Baseline):
    """Computes greedy rollout if 'num_rollouts' is -1 or sampled rollout
    if 'num_rollout' > 0."""
    def __init__(self, num_rollouts=-1, **kwargs):
        super().__init__(**kwargs)
        if num_rollouts == -1:
            self.strategy = 'greedy'
            self.num_rollouts = 1
        elif num_rollouts > 0:
            self.strategy = 'sampled'
            self.num_rollouts = num_rollouts
        else:
            raise TypeError("{} is not a valid number of rollouts, try with -1 for 'greedy' or 1, 2, 3, etc. for 'sampled'.".format(num_rollouts))

    def forward(self, formula, policy_network, num_variables):
        num_sats = []
        for i in range(self.num_rollouts):
            assignment = utils.sampling_assignment(policy_network, num_variables, self.strategy)
            # ::assigment:: [seq_len]
            _, num_sat, _ = utils.assignment_verifier(formula, assignment=assignment)
            num_sats.append(num_sat)
        baseline = torch.tensor(num_sat, dtype=float).mean()

        return baseline