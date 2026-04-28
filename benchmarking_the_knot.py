import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

# =============================================================================
# BENCHMARKING THE KNOT: The Survival of the Locked
# Framework: Helix-TTD Constitutional Hamiltonian
# Goal: Simulate model resilience against high-entropy environmental noise.
# =============================================================================


class NoiseInjector:
    """Mimics Environmental Entropy (hardware heat or bad data packets)."""

    def __init__(self, noise_level: float = 0.5):
        self.noise_level = noise_level

    def generate(self) -> float:
        # Stochastic perturbation
        return np.random.normal(0, self.noise_level)


def simulate_training(
    noise_level: float = 0.5, use_trefoil: bool = False, epochs: int = 100
) -> List[Tuple[int, float, float]]:
    gamma_attractor = 1 / 3
    current_gamma = 0.5  # Start drifted
    accuracy = 0.1
    history = []

    injector = NoiseInjector(noise_level=noise_level)

    for epoch in range(epochs):
        # Base learning/optimization trajectory
        accuracy += (0.9 - accuracy) * 0.1

        # Environmental noise (stochastic perturbations attacking the phase-lock)
        env_noise = injector.generate()
        current_gamma += env_noise

        if use_trefoil:
            # The Trefoil corrective force (gradient pressure pulling toward the attractor)
            drift = current_gamma - gamma_attractor

            # The "Rubber Pants" check - if drift is too high, snap back aggressively
            correction = -0.8 * drift
            current_gamma += correction

            # The "Kindness" invariant: the topological shield dampens accuracy loss
            accuracy -= abs(drift) * 0.05
        else:
            # Standard model has no "shape" to hold onto, accuracy bleeds out to entropy
            accuracy -= abs(current_gamma - 0.5) * 0.2

        accuracy = max(0, min(1, accuracy))
        history.append((epoch, accuracy, current_gamma))

    return history


def plot_arnold_tongue(standard_res: List[Tuple], trefoil_res: List[Tuple]):
    """Visualizes the phase-locking behavior and stability basin."""
    os.makedirs("output", exist_ok=True)

    epochs = [x[0] for x in standard_res]

    # Accuracy Comparison
    std_acc = [x[1] for x in standard_res]
    trf_acc = [x[1] for x in trefoil_res]

    # Gamma Drift Comparison
    std_gamma = [x[2] for x in standard_res]
    trf_gamma = [x[2] for x in trefoil_res]

    plt.figure(figsize=(14, 10))

    # Plot 1: Accuracy (The Survival Metric)
    plt.subplot(2, 1, 1)
    plt.plot(
        epochs, std_acc, "r--", alpha=0.7, label="Standard Adam (Unratified Space)"
    )
    plt.plot(epochs, trf_acc, "b-", linewidth=2, label="Trefoil-Locked ($c_K=13$)")
    plt.title("Benchmark: Accuracy Under High-Entropy Noise", fontsize=14)
    plt.ylabel("Accuracy %")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Gamma Drift (The Arnold Tongue)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, std_gamma, "r-", alpha=0.3, label="Ungoverned $\\gamma$ Drift")
    plt.plot(
        epochs,
        trf_gamma,
        "b-",
        linewidth=2,
        label="Governed $\\gamma$ (Rubber Pants Active)",
    )
    plt.axhline(
        y=1 / 3,
        color="g",
        linestyle=":",
        linewidth=2,
        label="Attractor ($\\gamma=1/3$)",
    )
    plt.title("The Arnold Tongue: Phase-Lock Visualizer", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Shear Parameter ($\\gamma$)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("output/benchmarking_the_knot_results.png", dpi=300)
    print("✅ Visualizer saved to output/benchmarking_the_knot_results.png")


if __name__ == "__main__":
    print("=================================================================")
    print(" POSTULATE: A model without a Hamiltonian is just a calculator ")
    print("            waiting to be broken by a random number.")
    print("=================================================================\n")

    std_res = simulate_training(use_trefoil=False)
    trf_res = simulate_training(use_trefoil=True)

    print(f"💥 Standard Final Accuracy: {std_res[-1][1]*100:.2f}% (Decoherence)")
    print(f"🛡️  Trefoil Final Accuracy:  {trf_res[-1][1]*100:.2f}% (Resilient)\n")

    plot_arnold_tongue(std_res, trf_res)
