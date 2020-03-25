"""Unit tests for MPS simulations."""

import numpy as np

from mpsim import simulate


def test_fidelity():
    for n in range(5, 20, 5):
        for d in range(10, 20, 5):
            wavefunction = simulate(
                nqubits=10, depth=10, fraction=1
            ).wavefunction
            assert np.isclose(np.linalg.norm(wavefunction), 1.0)


def test_seed():
    n = 10
    d = 10
    f = 1
    seed = 123
    wf1 = simulate(n, d, f, seed=seed).wavefunction
    wf2 = simulate(n, d, f, seed=seed).wavefunction
    wf3 = simulate(n, d, f, seed=1).wavefunction
    assert wf1 is not wf2
    assert np.allclose(wf1, wf2)
    assert not np.allclose(wf1, wf3)

