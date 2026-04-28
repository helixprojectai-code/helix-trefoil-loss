import pytest
import math
from trefoil_scheduler import TrefoilScheduler


def test_scheduler_initialization():
    scheduler = TrefoilScheduler(
        initial_gamma=1.0, target_gamma=1 / 3, total_epochs=100
    )
    assert scheduler.get_gamma() == 1.0
    assert scheduler.target_gamma == 1 / 3


def test_linear_annealing():
    scheduler = TrefoilScheduler(
        initial_gamma=1.0, target_gamma=1 / 3, total_epochs=10, anneal_strategy="linear"
    )

    # Halfway point (Epoch 5)
    gamma_half = scheduler.step(epoch=5)
    expected_half = 1.0 - 0.5 * (1.0 - 1 / 3)
    assert math.isclose(gamma_half, expected_half, rel_tol=1e-5)

    # Endpoint (Epoch 10)
    gamma_end = scheduler.step(epoch=10)
    assert math.isclose(gamma_end, 1 / 3, rel_tol=1e-5)

    # Past Endpoint (Epoch 15) should lock at target
    gamma_past = scheduler.step(epoch=15)
    assert math.isclose(gamma_past, 1 / 3, rel_tol=1e-5)


def test_cosine_annealing():
    scheduler = TrefoilScheduler(
        initial_gamma=1.0,
        target_gamma=1 / 3,
        total_epochs=100,
        anneal_strategy="cosine",
    )

    # Endpoint
    gamma_end = scheduler.step(epoch=100)
    assert math.isclose(gamma_end, 1 / 3, rel_tol=1e-5)
