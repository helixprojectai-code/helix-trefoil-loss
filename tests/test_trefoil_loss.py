import torch
from trefoil_loss import TrefoilLoss

def test_initialization():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    assert criterion.target_gamma == 1/3
    assert criterion.topological_multiplier > 0

def test_phase_locked_loss():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    
    total_loss = criterion(base_loss, current_gamma=1/3)
    assert torch.isclose(total_loss, base_loss, atol=1e-5)

def test_drift_penalty():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    
    total_loss = criterion(base_loss, current_gamma=0.45)
    assert total_loss > base_loss

def test_trace_invariant_penalty():
    criterion = TrefoilLoss(target_gamma=1/3, protection_factor=13, trace_weight=0.1)
    base_loss = torch.tensor(2.5)
    
    perfect_weight = torch.ones(4, 4)
    total_loss_perfect = criterion(base_loss, current_gamma=1/3, weight_tensors=[perfect_weight])
    assert torch.isclose(total_loss_perfect, base_loss, atol=1e-5)
    
    drifting_weight = torch.ones(4, 4) * 2.0
    total_loss_drift = criterion(base_loss, current_gamma=1/3, weight_tensors=[drifting_weight])
    assert total_loss_drift > base_loss
