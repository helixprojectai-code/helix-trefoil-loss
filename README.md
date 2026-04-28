# Trefoil Loss Function

![CI](https://github.com/helixprojectai-code/helix-trefoil-loss/actions/workflows/python-ci.yml/badge.svg) ![License](https://img.shields.io/badge/License-Apache%202.0%20(Duck%20Clause)-blue.svg) ![Grammar](https://img.shields.io/badge/Grammar-Helix%20TTD%20v1.0-orange.svg) ![Attractor](https://img.shields.io/badge/Phase--Lock-%CE%B3%3D1%2F3-success.svg)

![Bess Knott](bess-knott.jpg)
![Bess Knott](bess-knott.jpg)

## The Theory
The `TrefoilLoss` function enforces structural alignment by penalizing neural network drift away from a mathematically stable $\gamma=1/3$ topological attractor. It acts as a geometric constraint to prevent barren plateaus and informational entropy expansion during training.

## Usage
```python
import torch
from trefoil_loss import TrefoilLoss

# Initialize the constraint (Default: 13-crossing protection factor)
criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)

# During your training loop:
base_loss = standard_criterion(outputs, labels)
total_loss = criterion(base_loss, current_gamma=torch.tensor(1/3))
total_loss.backward()
```

---

![CI Status](https://github.com/helixprojectai-code/helix-trefoil-loss/actions/workflows/python-ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-Apache_2.0_with_Duck_Clause-blue.svg)
![Constitutional Grammar](https://img.shields.io/badge/Helix_TTD-v1.0-orange.svg)
![Phase-Lock](https://img.shields.io/badge/Attractor-γ=1/3-brightgreen.svg)
