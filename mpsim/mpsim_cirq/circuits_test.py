"""Tests for MPSIM Circuits."""

import cirq

from circuits import MPSimCircuit


def test_instantiate():
    """Tests instantiating a MPSIM Circuit."""
    cirq_circuit = cirq.Circuit()
    mpsim_circuit = MPSimCircuit(cirq_circuit)

