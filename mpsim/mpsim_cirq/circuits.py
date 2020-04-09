"""Defines mpsim circuits as extensions of Cirq circuits."""

from typing import Iterable, List, Tuple

import numpy as np

import cirq
from tensornetwork import Node


class CannotConvertToMPSOperation(Exception):
    pass


class MPSOperation:
    """Defines an operation for MPS Simulators."""
    def __init__(
            self,
            node: Node,
            qudit_indices: Tuple[int],
            qudit_dimension: int = 2
    ) -> None:
        """Constructor for an MPS Instruction.

        Args:
            node: TensorNetwork node object representing a gate to apply.
                   See Notes below.
            qudit_indices: Indices of qubits to apply the gate to.
            qudit_dimension: Dimension of qudit(s) to which the MPS Operation
                              is applied. Default value is 2 (for qubits).

        Notes:
            Conventions for gates and edges.
                TODO: Add explanation on edge conventions.
        """
        self._node = node
        self._qudit_indices = tuple(qudit_indices)
        self._qudit_dimension = int(qudit_dimension)

    @property
    def qudit_indices(self) -> Tuple[int]:
        """Returns the indices of the qudits that the MPS Operation acts on."""
        return self._qudit_indices

    @property
    def qudit_dimension(self) -> int:
        """Returns the dimension of the qudits the MPS Operation acts on."""
        return self._qudit_dimension

    @property
    def num_qudits(self) -> int:
        """Returns the number of qubits the MPS Operation acts on."""
        return len(self._qudit_indices)

    def is_valid(self) -> bool:
        """Returns True if the MPS Operation is valid, else False.

        A valid MPS Operation meets the following criteria:
            (1) Tensor of gate is d x d where d = qudit dimension.
            (2) Tensor has 2n free edges where n = number of qudits.
            (3) All tensor edges are free edges.
        """
        d = self._qudit_dimension
        if not self._node.tensor.shape == tuple([self._qudit_dimension] * d):
            return False

        if not len(self._node.get_all_edges()) == 2 * self.num_qudits:
            return False

        if self._node.has_nondangling_edge():
            return False

        return True

    def is_unitary(self) -> bool:
        """Returns True if the MPS Operation is unitary, else False.

        An MPS Operation is unitary if its gate tensor U is unitary, i.e. if
        U^dag @ U = U @ U^dag = I.
        """
        return cirq.equal_up_to_global_phase(
            self._node.tensor.conj().T @ self._node.tensor,
            np.identity(self._qudit_dimension ** self.num_qudits)
        )

    def is_single_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on a single qudit."""
        return self.num_qudits == 1

    def is_two_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on two qudits."""
        return self.num_qudits == 2

    def __str__(self):
        return f"Tensor {self._node.name} on qudits {self._qudit_indices}."


# TODO: Is this class necessary? The main functionality needed is to convert
#  the circuit into a list of operations that the MPS object can implement.
class MPSimCircuit(cirq.Circuit):
    """Defines MPS Circuits which extend cirq.Circuits and can be simulated by
    an MPS Simulator.
    """
    def __init__(
        self, 
        cirq_circuit: cirq.Circuit,
        device: cirq.devices = cirq.devices.UNCONSTRAINED_DEVICE
    ) -> None:
        """Constructor for MPSimCircuit.

        Args:
            cirq_circuit: Cirq circuit to create an MPS Sim circuit from.
            device: Device the circuit runs on.
        """
        # TODO: Check that device is one-dimensional, as required for MPS.
        super().__init__(cirq_circuit, device=device)
        self._mps_operations = self._translate_to_mps_operations()
        self._index_for_qudit = {
            i: qubit for i, qubit in enumerate(sorted(self.all_qubits()))
        }

    def _indices_for_qudits(self, qudits: Iterable[cirq.Qid]):
        """Yields a generator of indices for the given qubits.

        Args:
            qudits: Qudits in the circuit to get the indices for.
        """
        for qudit in qudits:
            yield self._index_for_qudit[qudit]

    def _convert_gate_operation_to_mps_operation(
            self,
            operation: cirq.GateOperation
    ) -> MPSOperation:
        """Converts a gate operation to an MPS operation and returns it.

        Args:
            operation: A valid cirq.GateOperation, or any child classes.
        """
        num_qudits = len(operation.qubits)
        qudit_dimension = 2
        qudit_indices = self._indices_for_qudits(operation.qubits)

        if not operation._has_unitary_():
            raise CannotConvertToMPSOperation(
                f"Cannot convert operation {operation} into an MPS Operation"
                " because the operation does not have a unitary."
            )
        tensor = operation._unitary_()
        tensor = np.reshape(tensor, newshape=[qudit_dimension] * num_qudits)
        node = Node(tensor)

        return MPSOperation(node, tuple(qudit_indices), qudit_dimension)

    # TODO: Should this keep the same moment/operation circuit structure?
    #  Or should it just be one-dimensional?
    def _translate_to_mps_operations(self) -> List[MPSOperation]:
        """Appends all operations in a circuit to MPS operations."""
        all_mps_operations = []
        for (moment_index, moment) in enumerate(self):
            for operation in moment:
                all_mps_operations.append(
                    self._convert_gate_operation_to_mps_operation(operation)
                )
        return all_mps_operations

    # TODO: Every time a gate is added to the circuit, also add it to
    #  self._mpsim_operations. E.g.,
    #  mpsim_circuit.append([some new gates])
    #  or
    #  mpsim_circuit.insert([some gates at some location])
    #  Should update mpsim_circuit._mpsim_operations
