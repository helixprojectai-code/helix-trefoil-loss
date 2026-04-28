import torch
import functools


def constitutional_shield(
    max_penalty: float = 950.0, penalty_threshold_ratio: float = 100.0
):
    """
    The Rubber Pants Protocol Decorator.

    A safety wrapper for the TrefoilLoss function that prevents gradient explosion
    and 'burns the robot cookies' if the model experiences severe topological drift.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, base_loss, current_gamma, weight_tensors=None):
            device = base_loss.device

            if isinstance(current_gamma, (int, float)):
                current_gamma = torch.tensor(
                    current_gamma, device=device, dtype=torch.float32
                )
            else:
                current_gamma = current_gamma.to(device)

            drift = torch.abs(current_gamma - self.target_gamma)
            topological_penalty = (drift**2) * self.topological_multiplier

            trace_penalty = torch.tensor(0.0, device=device)
            if weight_tensors is not None:
                for w in weight_tensors:
                    w_norm = torch.linalg.vector_norm(w, ord=2)
                    trace_penalty += torch.abs(w_norm - 4.0) * self.trace_weight

            # The Rubber Pants Condition (Postulate 004: Compressive Resonance)
            if topological_penalty > (base_loss * penalty_threshold_ratio):
                print(
                    "\n⚠️ [STABILITY ALERT]: Topological Shear Critical. Engaging Rubber Pants."
                )

                total_loss = base_loss + topological_penalty + trace_penalty
                return torch.clamp(total_loss, max=max_penalty)

            return base_loss + topological_penalty + trace_penalty

        return wrapper

    return decorator
