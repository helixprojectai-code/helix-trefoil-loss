import torch
import pytest
from trefoil_loss import TrefoilLoss

def test_initialization():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    assert criterion.target_gamma == 1/3
    assert criterion.topological_multiplier > 0

def test_phase_locked_loss():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    
    # At exactly the target_gamma, topological penalty should be zero
    total_loss = criterion(base_loss, current_gamma=1/3)
    
    # Allow for floating point precision differences
    assert torch.isclose(total_loss, base_loss, atol=1e-5)

def test_drift_penalty():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    
    total_loss = criterion(base_loss, current_gamma=0.45)
    
    # Loss should be significantly higher than base loss due to penalty
    assert total_loss > base_loss

def test_trace_invariant_penalty():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13, trace_weight=0.1)
    base_loss = torch.tensor(2.5)
    
    # Create a dummy weight tensor whose L2 norm is exactly 4.0
    # A tensor of shape (4, 4) filled with 1.0 has L2 norm = sqrt(16) = 4.0
    perfect_weight = torch.ones(4, 4)
    
    # Trace penalty should be zero
    total_loss_perfect = criterion(base_loss, current_gamma=1/3, weight_tensors=[perfect_weight])
    assert torch.isclose(total_loss_perfect, base_loss, atol=1e-5)
    
    # Create a drifting weight tensor (norm != 4.0)
    drifting_weight = torch.ones(4, 4) * 2.0
    
    total_loss_drift = criterion(base_loss, current_gamma=1/3, weight_tensors=[drifting_weight])
    assert total_loss_drift > base_loss
