"""Unit tests for gates."""

import numpy as np
import pytest

from mpsim.gates import (
    computational_basis_projector,
    is_projector,
    is_unitary,
    igate,
    hgate,
    xgate,
    ygate,
    zgate,
    cnot,
    cphase,
    haar_random_unitary
)


def test_is_unitary():
    """Tests that common qubit gates are unitary."""
    for gate in (igate(), hgate(), xgate(), ygate(), zgate(), cnot()):
        assert is_unitary(gate)

    for exp in np.linspace(start=0, stop=2 * np.pi, num=100):
        assert is_unitary(cphase(exp))


def test_qubit_pi0_projector():
    """Tests correctness for |0><0| on a qubit."""
    pi0 = computational_basis_projector(state=0)
    correct_tensor = np.array([[1., 0.], [0., 0.]])
    assert np.array_equal(pi0.tensor, correct_tensor)
    assert is_projector(pi0)
    assert not is_unitary(pi0)
    assert pi0.__str__() == "|0><0|"


def test_qubit_pi1_projector():
    """Tests correctness for |0><0| on a qubit."""
    pi1 = computational_basis_projector(state=1)
    correct_tensor = np.array([[0., 0.], [0., 1.]])
    assert np.array_equal(pi1.tensor, correct_tensor)
    assert is_projector(pi1)
    assert not is_unitary(pi1)
    assert pi1.__str__() == "|1><1|"


def test_invalid_projectors():
    """Tests exceptions are raised for invalid projectors."""
    # State must be positive
    with pytest.raises(ValueError):
        computational_basis_projector(state=-1)

    # Dimension must be positive
    with pytest.raises(ValueError):
        computational_basis_projector(state=2, dim=-1)

    # State must be less than dimension
    with pytest.raises(ValueError):
        computational_basis_projector(state=10, dim=8)


def test_qutrit_projectors():
    """Tests correctness for projectors on qutrits."""
    dim = 3
    for state in (0, 1, 2):
        projector = computational_basis_projector(state, dim)
        correct_tensor = np.zeros((dim, dim))
        correct_tensor[state, state] = 1.
        assert np.array_equal(projector.tensor, correct_tensor)
        assert is_projector(projector)
        assert not is_unitary(projector)
        assert projector.__str__() == f"|{state}><{state}|"


def test_haar_random_unitary():
    """Tests single-qubit and two-qubit Haar random unitaries."""
    for n in [2, 3, 4]:
        for d in [2, 3, 5]:
            gate = haar_random_unitary(nqudits=n, qudit_dimension=d, seed=1)
            assert is_unitary(gate)
