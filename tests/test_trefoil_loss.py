import pytest
import torch
import numpy as np
from trefoil_loss import TrefoilLoss

def test_initialization():
    """Test that the module initializes correctly with default and custom parameters."""
    loss_fn = TrefoilLoss()
    assert loss_fn.target_gamma == 1/3
    assert loss_fn.P > 0
    
    loss_fn_custom = TrefoilLoss(target_gamma=0.5, protection_factor=7)
    assert loss_fn_custom.target_gamma == 0.5
    expected_P = np.exp(2.302585 * 7)
    assert np.isclose(loss_fn_custom.P, expected_P)

def test_phase_locked_penalty():
    """Test that the topological penalty is zero when phase-locked at gamma=1/3."""
    loss_fn = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    current_gamma = torch.tensor(1/3)
    total_loss = loss_fn(base_loss, current_gamma)
    assert torch.allclose(total_loss, base_loss, atol=1e-6)

def test_drift_penalty():
    """Test that the topological penalty applies correctly when drifting from gamma=1/3."""
    loss_fn = TrefoilLoss(target_gamma=1/3, protection_factor=13)
    base_loss = torch.tensor(2.5)
    current_gamma = torch.tensor(0.4)
    total_loss = loss_fn(base_loss, current_gamma)
    assert total_loss > base_loss

def test_trace_invariant_penalty():
    """Test that the trace invariant penalty functions correctly."""
    loss_fn = TrefoilLoss(target_gamma=1/3, protection_factor=3)
    base_loss = torch.tensor(1.0)
    current_gamma = torch.tensor(1/3)
    perfect_weights = torch.tensor([2.0, 2.0, 2.0, 2.0])
    loss_perfect = loss_fn(base_loss, current_gamma, weight_tensor=perfect_weights)
    assert torch.allclose(loss_perfect, base_loss, atol=1e-6)
    
    drifting_weights = torch.tensor([10.0, 10.0])
    loss_drifting = loss_fn(base_loss, current_gamma, weight_tensor=drifting_weights)
    assert loss_drifting > base_loss
