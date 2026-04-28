import torch
import torch.nn as nn
import numpy as np


class TrefoilLoss(nn.Module):
    def __init__(self, target_gamma=1 / 3, protection_factor=13):
        """
        Initializes the Trefoil Topological Constraint.
        Args:
            target_gamma (float): The stability attractor (default 1/3).
            protection_factor (int): The knot crossing complexity (c_K).
        """
        super(TrefoilLoss, self).__init__()
        self.target_gamma = target_gamma
        self.P = np.exp(2.302585 * protection_factor)

    def forward(self, base_loss, current_gamma, weight_tensor=None):
        """
        Calculates the topological penalty.
        Args:
            base_loss (Tensor): The standard loss (e.g., CrossEntropy).
            current_gamma (float/Tensor): The shear parameter or learning rate proxy.
            weight_tensor (Tensor, optional): The output layer weights for Trace enforcement.
        """
        # The Topological Penalty (Drift from the Attractor)
        drift = torch.abs(current_gamma - self.target_gamma)
        topological_penalty = (drift**2) * (self.P / 1e6)

        # The Trace Invariant (|Tr| = 4)
        trace_penalty = 0.0
        if weight_tensor is not None:
            weight_norm = torch.norm(weight_tensor, p=2)
            trace_penalty = torch.abs(weight_norm - 4.0) * 0.1

        return base_loss + topological_penalty + trace_penalty
