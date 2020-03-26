"""Unit tests for MPS simulations."""

import numpy as np

from mpsim import simulate


def test_fidelity_with_fraction():
    for n in range(5, 20, 5):
        for d in range(10, 20, 5):
            wavefunction = simulate(
                nqubits=10, depth=10, fraction=1
            ).wavefunction
            assert np.isclose(np.linalg.norm(wavefunction), 1.0)


def test_fidelity_with_maxsvals():
    for n in range(5, 20, 5):
        for d in range(10, 20, 5):
            wavefunction = simulate(
                nqubits=10, depth=10, maxsvals=2**n,
            ).wavefunction
            assert np.isclose(np.linalg.norm(wavefunction), 1.0)


def test_seed_with_fraction():
    n = 10
    d = 10
    f = 1
    seed = 123
    wf1 = simulate(n, d, seed=seed, fraction=f).wavefunction
    wf2 = simulate(n, d, seed=seed, fraction=f).wavefunction
    wf3 = simulate(n, d, seed=1, fraction=f).wavefunction
    assert wf1 is not wf2
    assert np.allclose(wf1, wf2)
    assert not np.allclose(wf1, wf3)


def test_seed_with_maxsvals():
    n = 10
    d = 10
    f = 1
    seed = 123
    wf1 = simulate(n, d, seed=seed, fraction=f).wavefunction
    wf2 = simulate(n, d, seed=seed, fraction=f).wavefunction
    wf3 = simulate(n, d, seed=1, fraction=f).wavefunction
    assert wf1 is not wf2
    assert np.allclose(wf1, wf2)
    assert not np.allclose(wf1, wf3)

