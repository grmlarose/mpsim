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
