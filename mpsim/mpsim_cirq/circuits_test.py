"""Tests for MPSIM Circuits."""

import cirq

from mpsim.mpsim_cirq.circuits import MPSimCircuit


def test_instantiate():
    """Tests instantiating an mpsim circuit."""
    cirq_circuit = cirq.Circuit()
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    assert len(list(mpsim_circuit.all_qubits())) == 0
    assert len(list(mpsim_circuit.all_operations())) == 0
