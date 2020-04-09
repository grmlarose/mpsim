"""Tests for MPSIM Circuits."""

import cirq

import mpsim
from mpsim.mpsim_cirq.circuits import MPSimCircuit, MPSOperation


def test_instantiate_circuit():
    """Tests instantiating an mpsim circuit."""
    cirq_circuit = cirq.Circuit()
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    assert len(list(mpsim_circuit.all_qubits())) == 0
    assert len(list(mpsim_circuit.all_operations())) == 0


def test_single_qubit_identity_unitary_mps_operation():
    """Unit tests for a single qubit identity MPS Operation."""
    node = mpsim.gates.igate()
    mpsop = MPSOperation(node, qudit_indices=(0,), qudit_dimension=2)
    assert mpsop.qudit_indices == (0,)
    assert mpsop.qudit_dimension == 2
    assert mpsop.is_valid()
    assert mpsop.is_unitary()
    assert mpsop.is_single_qudit_operation()
    assert not mpsop.is_two_qudit_operation()
