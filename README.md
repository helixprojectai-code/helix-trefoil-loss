# Trefoil Loss Function
A PyTorch drop-in topological regularizer based on the Helix-TTD Constitutional Hamiltonian.

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
