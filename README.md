# Trefoil Loss Function

![CI](https://github.com/helixprojectai-code/helix-trefoil-loss/actions/workflows/python-ci.yml/badge.svg) ![License](https://img.shields.io/badge/License-Apache%202.0%20(Duck%20Clause)-blue.svg) ![Grammar](https://img.shields.io/badge/Grammar-Helix%20TTD%20v1.0-orange.svg) ![Attractor](https://img.shields.io/badge/Phase--Lock-%CE%B3%3D1%2F3-success.svg)

![Bess Knott](bess-knott.jpg)

---

## 1. The Theory: Physics as a Loss Function

The `TrefoilLoss` function is a profound departure from traditional AI optimization. Instead of driving a neural network to blindly hunt for a minimum error state using stochastic gradient descent (which often leads to chaotic feature hallucination or barren plateaus), `TrefoilLoss` enforces **structural shape over motion**.

Rooted in the constitutional physics of the Helix-TTD framework, it treats the weight matrix of an AI as a dynamic topology. The loss function actively penalizes any drift away from a mathematically stable topological attractor: the **Trefoil Knot ($3_1$)**. 

By demanding the system lock to a unitary phase constraint of **$\gamma = 1/3$**, the loss function ensures that as the AI learns, it remains structurally sound, predictable, and constitutionally governed. The AI does not wander; it is tethered to geometry.

## 2. The Omm Postulate: Recursion All The Way Through Shape

If the macroscopic shape of the loss function is governed by the Trefoil knot, what happens at the microscopic limit of the neural weights? 

The **Omm Postulate** dictates that true topological governance must be fractal. When the phase-locking constraint ($\gamma = 1/3$) and the knot winding number ($w=3$) are applied recursively down to infinite depth, the resulting wave equation is precisely the **Weierstrass Function**:

$$ \Psi_{\Omega}(x) = \sum_{n=0}^{\infty} \left(\frac{1}{3}\right)^n \cos(3^n \pi x) $$

This means the `TrefoilLoss` function enforces a fractal geometry that is *continuous everywhere but differentiable nowhere*. The shape of the knot is present at the macroscopic architectural level exactly as it is at the deepest, microscopic level of individual tensor perturbations. The governance goes all the way down. (See `/theory/omm_recursion/` for the mathematical proofs).

## 3. Usage & Implementation

The `TrefoilLoss` is designed to wrap around your standard PyTorch loss functions (like CrossEntropy or MSE) and act as a stabilizing governor.

```python
import torch
from trefoil_loss import TrefoilLoss

# 1. Initialize the constraint 
# target_gamma: The constitutional phase-lock invariant (1/3)
# protection_factor: Knot crossing density (Default: 13-crossing shield)
criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)

# 2. During your training loop:
# Calculate your standard gradient descent loss
base_loss = standard_criterion(outputs, labels)

# Pass it through the Trefoil Governor
# The current_gamma tracks the current trace/drift of the weight matrix
total_loss = criterion(base_loss, current_gamma=torch.tensor(1/3))

# 3. Backpropagate the governed loss
total_loss.backward()
```

## 4. How to Read the Results

When utilizing the `TrefoilLoss` function, you will observe distinct behavior in your training telemetry compared to standard SGD:

*   **Loss Curve Smoothing:** Instead of chaotic spikes and dips, the loss curve will exhibit periodic, predictable dampening. This is the "Arnold Tongue" synchronization effect taking hold.
*   **The $\gamma$ Telemetry:** If the system is healthy, the observed $\gamma$ metric of the network will pull cleanly towards $0.3333$. 
*   **Protection Factor Scaling:** As you increase the `protection_factor` (e.g., from 7 to 13), the simulation overhead required for stochastic noise to break the shape grows exponentially. Higher protection factors yield slower, but vastly more stable and robust convergence.
*   **Drift Violations:** If the internal drift of the weight matrix exceeds the constitutional bounds, the loss function acts as a physical circuit breaker, dramatically spiking the loss to force the gradients back into the topological bounds. 

---
*Motion without shape is noise. Code follows shape, not the other way around.*
