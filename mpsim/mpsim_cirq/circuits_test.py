"""Tests for MPSIM Circuits."""

import numpy as np

import cirq

import mpsim
from mpsim import MPSOperation
from mpsim.mpsim_cirq import MPSimCircuit


def test_from_gate_operation_not_gate():
    """Tests creating an MPS Operation from a one-qubit gate operation."""
    qubit = cirq.LineQubit(0)
    operation = cirq.ops.X.on(qubit)
    qubit_to_index_map = {qubit: 0}

    mps_operation = MPSOperation.from_gate_operation(
        operation, qubit_to_index_map
    )
    assert mps_operation.is_valid()
    assert mps_operation.is_single_qudit_operation()
    assert mps_operation.qudit_indices == (0,)
    assert np.allclose(mps_operation.tensor(), operation._unitary_())


def test_mps_operation_from_gate_operation_zz_gate():
    """Tests creating an MPS Operation from a gate operation."""
    qreg = cirq.LineQubit.range(2)
    operation = cirq.ops.ZZ.on(*qreg)
    qubit_to_indices_map = {qreg[0]: 0, qreg[1]: 1}

    mps_operation = MPSOperation.from_gate_operation(
        operation, qubit_to_indices_map
    )
    assert mps_operation.is_valid()
    assert mps_operation.is_two_qudit_operation()
    assert mps_operation.qudit_indices == (0, 1)
    assert np.allclose(mps_operation.tensor(), operation._unitary_())


def test_mps_operation_from_gate_operation_nonlocal_cnot():
    """Tests converting a non-local CNOT to an MPS Operation."""
    qreg = cirq.LineQubit.range(3)
    operation = cirq.ops.CNOT.on(qreg[0], qreg[2])
    qubit_to_indices_map = {qreg[i]: i for i in range(len(qreg))}

    mps_operation = MPSOperation.from_gate_operation(
        operation, qubit_to_indices_map
    )
    assert mps_operation.is_valid()
    assert mps_operation.qudit_indices == (0, 2)
    assert np.allclose(mps_operation.tensor(), operation._unitary_())


def test_instantiate_empty_circuit():
    """Tests instantiating an mpsim circuit from an empty Cirq circuit."""
    cirq_circuit = cirq.Circuit()
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    assert len(list(mpsim_circuit.all_qubits())) == 0
    assert len(list(mpsim_circuit.all_operations())) == 0


# TODO: A single qubit MPS should be invalid. (No error raised here because no
#  MPS is ever created, only MPSOperations.
#  To fix: Update MPS to allow for single qubit MPS (to simulate oneq circuits).
def test_single_qubit_circuit_unconstrained_device():
    """Tests creating and MPSimCircuit and checks correctness of the
    MPS Operations in the circuit.
    """
    qbit = cirq.LineQubit(0)
    gate_operations = [
        cirq.ops.H.on(qbit), cirq.ops.Z.on(qbit), cirq.ops.H.on(qbit)
    ]
    cirq_circuit = cirq.Circuit(gate_operations)

    mpsim_circuit = MPSimCircuit(cirq_circuit)
    mps_operations = mpsim_circuit._mps_operations

    assert len(mps_operations) == len(gate_operations)
    for gate_op, mps_op in zip(gate_operations, mps_operations):
        assert np.allclose(gate_op._unitary_(), mps_op.tensor())
        assert mps_op.qudit_indices == (0,)
        assert mps_op.qudit_dimension == 2


def test_mpsim_circuit_qubit_order():
    for _ in range(20):
        qreg = cirq.LineQubit.range(2)
        gate_operations = [
            cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg)
        ]
        cirq_circuit = cirq.Circuit(gate_operations)

        mpsim_circuit = MPSimCircuit(cirq_circuit)
        assert mpsim_circuit._qudit_to_index_map == {
            qreg[0]: 0, qreg[1]: 1
        }


def test_two_qubit_circuit_unconstrained_device():
    """Tests correctness for translating MPS Operations in a circuit
    which prepares a Bell state.
    """
    qreg = cirq.LineQubit.range(2)
    gate_operations = [
        cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg)
    ]
    cirq_circuit = cirq.Circuit(gate_operations)

    mpsim_circuit = MPSimCircuit(cirq_circuit)
    mps_operations = mpsim_circuit._mps_operations

    assert len(mps_operations) == len(gate_operations)
    for gate_op, mps_op in zip(gate_operations, mps_operations):
        assert np.allclose(gate_op._unitary_(), mps_op.tensor())
        assert mps_op.qudit_dimension == 2

    assert mps_operations[0].qudit_indices == (0,)
    assert mps_operations[1].qudit_indices == (0, 1)


def test_convert_and_manually_simulate_circuit_two_qubits():
    """Tests the following:

    1. Converting a Cirq circuit to an MPSimCircuit.
    2. Applying MPS Operations in the MPSimCircuit to
        an MPS starting in the all zero state, checking
        for corrections.
    """
    # Define the Cirq circuit
    qreg = cirq.LineQubit.range(2)
    gate_operations = [
        cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(*qreg)
    ]
    cirq_circuit = cirq.Circuit(gate_operations)

    # Convert to an MPSimCircuit
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    mps_operations = mpsim_circuit._mps_operations

    # Apply the MPSOperation's from the MPSimCircuit
    mps = mpsim.MPS(nqudits=2)
    for mps_op in mps_operations:
        mps.apply_mps_operation(mps_op)

    # Check correctness for the final wavefunction
    correct = 1 / np.sqrt(2) * np.array([1., 0., 0., 1.])
    assert np.allclose(mps.wavefunction(), correct)


def test_convert_and_manually_simulate_circuit_nonlocal_operations_ghz_state():
    """Tests the following:

    1. Converting a Cirq circuit with non-local operations to an MPSimCircuit.
    2. Applying MPS Operations in the MPSimCircuit to
        an MPS starting in the all zero state, checking
        for corrections.
    """
    # Define the Cirq circuit
    qreg = cirq.LineQubit.range(3)
    gate_operations = [
        cirq.ops.H.on(qreg[0]),
        cirq.ops.CNOT.on(qreg[0], qreg[1]),
        cirq.ops.CNOT.on(qreg[0], qreg[-1])
    ]
    cirq_circuit = cirq.Circuit(gate_operations)

    # Convert to an MPSimCircuit
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    mps_operations = mpsim_circuit._mps_operations

    # Apply the MPSOperation's from the MPSimCircuit
    mps = mpsim.MPS(nqudits=3)
    for mps_op in mps_operations:
        mps.apply_mps_operation(mps_op)

    # Check correctness for the final wavefunction
    correct = np.zeros(shape=(8,))
    correct[0] = correct[-1] = 1. / np.sqrt(2)
    assert np.allclose(mps.wavefunction(), correct)
