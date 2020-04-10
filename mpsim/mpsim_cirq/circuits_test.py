"""Tests for MPSIM Circuits."""

import numpy as np

import cirq
import tensornetwork as tn

import mpsim
from mpsim.mpsim_cirq.circuits import MPSimCircuit, MPSOperation


def test_single_qubit_identity_mps_operation():
    """Unit tests for a single-qubit identity MPS Operation."""
    node = mpsim.gates.igate()
    mps_operation = MPSOperation(node, qudit_indices=(0,), qudit_dimension=2)
    assert mps_operation.qudit_indices == (0,)
    assert mps_operation.qudit_dimension == 2
    assert mps_operation.is_valid()
    assert mps_operation.is_unitary()
    assert mps_operation.is_single_qudit_operation()
    assert not mps_operation.is_two_qudit_operation()


def test_get_node_and_tensor_one_qubit_mps_operation():
    """Tests getting the node of a one-qubit MPS Operation."""
    np.random.seed(1)
    tensor = np.random.randn(2, 2)
    node = tn.Node(tensor)
    mps_operation = MPSOperation(node, qudit_indices=(0,), qudit_dimension=2)
    copy_node = mps_operation.node
    # TODO: How to check Node equality with tensornetwork?
    assert len(node.edges) == len(copy_node.edges)
    # assert node == copy_node
    copy_tensor = mps_operation.tensor()
    assert np.allclose(tensor, copy_tensor)


def test_two_qubit_cnot_mps_operation():
    """Unit tests for a two-qubit identity MPS Operation."""
    node = mpsim.gates.cnot()
    mps_operation = MPSOperation(node, qudit_indices=(0, 1), qudit_dimension=2)
    assert mps_operation.qudit_indices == (0, 1)
    assert mps_operation.qudit_dimension == 2
    assert not mps_operation.is_single_qudit_operation()
    assert mps_operation.is_two_qudit_operation()


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


def test_instantiate_empty_circuit():
    """Tests instantiating an mpsim circuit."""
    cirq_circuit = cirq.Circuit()
    mpsim_circuit = MPSimCircuit(cirq_circuit)
    assert len(list(mpsim_circuit.all_qubits())) == 0
    assert len(list(mpsim_circuit.all_operations())) == 0


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
