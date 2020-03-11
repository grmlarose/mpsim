"""Unit tests for inital MPS states."""

import pytest

import numpy as np
import tensornetwork as tn

import mps.mps as mps


def test_mps_one_qubit():
    """Ensures an error is raised if the number of qubits is less than two."""
    with pytest.raises(ValueError):
        mps.get_zero_state_mps(1)


def test_get_wavefunction_simple():
    """Tests getting the wavefunction of a simple MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=3)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    assert isinstance(wavefunction, np.ndarray)
    assert wavefunction.shape == (8,)
    correct = np.array([1.] + [0.] * 7, dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_correctness_of_initial_product_state_two_qubits():
    """Tests that the contracted MPS is indeed the all zero state for two qubits."""
    lq, _ = mps.get_zero_state_mps(nqubits=2)
    wavefunction_node = tn.contract(lq[1])
    wavefunction = np.reshape(wavefunction_node.tensor, newshape=(4,))
    correct = np.array([1, 0, 0, 0], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_correctness_of_initial_product_state():
    """Tests that the contracted MPS is indeed the all zero state for multiple qubits."""
    for n in range(3, 10):
        mpslist = mps.get_zero_state_mps(n)
        wavefunction = mps.get_wavefunction_of_mps(mpslist)
        correct = np.array([1] + [0] * (2**n - 1), dtype=np.complex64)
        assert np.array_equal(wavefunction, correct)


@pytest.mark.parametrize(["gate", "expected"],
                         [(mps.xgate(), mps.one_state),
                          (mps.hgate(), mps.plus_state),
                          (mps.zgate(), mps.zero_state),
                          (mps.igate(), mps.zero_state)])
def test_apply_oneq_gate_xgate(gate, expected):
    """Tests application of a single qubit gate to several MPS."""
    for n in range(2, 8):
        for j in range(n):
            mpslist = mps.get_zero_state_mps(n)
            mps.apply_one_qubit_gate(gate, j, mpslist)
            final_state = np.reshape(mpslist[j].tensor, newshape=(2,))
            assert np.array_equal(final_state, expected)


def test_apply_oneq_gate_to_all():
    """Tests correctness for final wavefunction after applying a NOT gate to all qubits in a two-qubit MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate_to_all(mps.xgate(), mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_apply_twoq_cnot_two_qubits():
    """Tests for correctness of final wavefunction after applying a CNOT to a two-qubit MPS."""
    # Check that CNOT|10> = |11>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|00> = |00>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    print(mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([1., 0., 0., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)
