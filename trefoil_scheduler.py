import math
import warnings
from typing import Optional


class TrefoilScheduler:
    """
    Dynamic Gamma Annealing Scheduler for the Trefoil Loss Function.

    Gradually tightens the topological constraint (gamma) from an initial
    exploratory state down to the mathematically stable 1/3 attractor.
    """

    def __init__(
        self,
        initial_gamma: float = 1.0,
        target_gamma: float = 1 / 3,
        total_epochs: int = 100,
        anneal_strategy: str = "cosine",
    ):
        """
        Args:
            initial_gamma: Starting shear parameter (loose constraint).
            target_gamma: Final topological attractor (strict constraint).
            total_epochs: Total number of epochs over which to anneal.
            anneal_strategy: 'linear', 'cosine', or 'step'.
        """
        self.initial_gamma = initial_gamma
        self.target_gamma = target_gamma
        self.total_epochs = total_epochs
        self.anneal_strategy = anneal_strategy
        self.current_epoch = 0
        self.current_gamma = initial_gamma

    def step(self, epoch: Optional[int] = None) -> float:
        """
        Updates and returns the current gamma value based on the epoch progress.
        Call this at the end of each training epoch.
        """
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        # Cap the epoch at total_epochs to lock at the target attractor
        progress = min(1.0, self.current_epoch / self.total_epochs)

        if self.anneal_strategy == "linear":
            self.current_gamma = self.initial_gamma - progress * (
                self.initial_gamma - self.target_gamma
            )

        elif self.anneal_strategy == "cosine":
            # Smooth cosine annealing curve
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            self.current_gamma = (
                self.target_gamma
                + (self.initial_gamma - self.target_gamma) * cosine_decay
            )

        elif self.anneal_strategy == "step":
            # Drops gamma by 50% of the remaining distance every 20% of epochs
            steps = progress // 0.2
            self.current_gamma = self.initial_gamma - (
                self.initial_gamma - self.target_gamma
            ) * (1 - (0.5**steps))

        else:
            warnings.warn(
                f"Unknown strategy {self.anneal_strategy}. Defaulting to linear."
            )
            self.current_gamma = self.initial_gamma - progress * (
                self.initial_gamma - self.target_gamma
            )

        # Floating point safety check
        self.current_gamma = max(self.target_gamma, self.current_gamma)

        return self.current_gamma

    def get_gamma(self) -> float:
        """Returns the current gamma value without stepping."""
        return self.current_gamma
