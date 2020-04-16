"""Defines mpsim circuits as extensions of Cirq circuits."""

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

import cirq
from tensornetwork import copy, Node


class CannotConvertToMPSOperation(Exception):
    pass


class MPSOperation:
    """Defines an operation which an MPS can execute."""
    def __init__(
            self,
            node: Node,
            qudit_indices: Tuple[int, ...],
            qudit_dimension: int = 2
    ) -> None:
        """Constructor for an MPS Operation.

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

    @staticmethod
    def from_gate_operation(
            operation: cirq.GateOperation,
            qudit_to_index_map: Dict[cirq.Qid, int]
    ) -> 'MPSOperation':
        """Constructs an MPS Operation from a gate operation.

        Args:
            operation: A valid cirq.GateOperation or any child class.
            qudit_to_index_map: Dictionary to map qubits to MPS indices.

        Raises:
            CannotConvertToMPSOperation
                If the gate operation does not have a _unitary_ method.
        """
        num_qudits = len(operation.qubits)
        qudit_dimension = 2
        qudit_indices = tuple(
            [qudit_to_index_map[qudit] for qudit in operation.qubits]
        )

        if not operation._has_unitary_():
            raise CannotConvertToMPSOperation(
                f"Cannot convert operation {operation} into an MPS Operation"
                " because the operation does not have a unitary."
            )
        tensor = operation._unitary_()
        tensor = np.reshape(
            tensor, newshape=[qudit_dimension] * qudit_dimension**num_qudits
        )
        node = Node(tensor)
        return MPSOperation(node, qudit_indices, qudit_dimension)

    @property
    def qudit_indices(self) -> Tuple[int, ...]:
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

    @property
    def node(self) -> Node:
        """Returns the node of the MPS Operation."""
        node_dict, _ = copy([self._node])
        return node_dict[self._node]

    def tensor(self, square: bool = True) -> np.ndarray:
        """Returns the tensor of the MPS Operation.

        Args:
            square: If True, the shape of the returned tensor is dim x dim where
                    dim is the qudit dimension raised to the number of qudits
                    that the MPS Operator acts on.
        """
        tensor = deepcopy(self._node.tensor)
        if square:
            dim = self._qudit_dimension ** self.num_qudits
            tensor = np.reshape(
                tensor, newshape=(dim, dim)
            )
        return tensor

    def is_valid(self) -> bool:
        """Returns True if the MPS Operation is valid, else False.

        A valid MPS Operation meets the following criteria:
            (1) Tensor of gate has shape (d, ..., d) where d is the qudit
                dimension and there are d^num_qudits entries in the tuple.
            (2) Tensor has 2n free edges where n = number of qudits.
            (3) All tensor edges are free edges.
        """
        d = self._qudit_dimension
        if not self._node.tensor.shape == tuple([d] * d ** self.num_qudits):
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
        return f"Tensor {self._node.name} on qudit(s) {self._qudit_indices}."


# TODO: Is this class necessary? The main functionality needed is to convert
#  the circuit into a list of operations that the MPS object can implement.
class MPSimCircuit(cirq.Circuit):
    """Defines MPS Circuits which extend cirq.Circuits and can be simulated by
    an MPS Simulator.
    """
    def __init__(
        self, 
        cirq_circuit: cirq.Circuit,
        device: cirq.devices = cirq.devices.UNCONSTRAINED_DEVICE,
        qubit_order: cirq.ops.QubitOrder = cirq.ops.QubitOrder.DEFAULT
    ) -> None:
        """Constructor for MPSimCircuit.

        Args:
            cirq_circuit: Cirq circuit to create an MPS Sim circuit from.
            device: Device the circuit runs on.
            qubit_order: Ordering of qubits.
        """
        # TODO: Check that device is one-dimensional, as required for MPS.
        super().__init__(cirq_circuit, device=device)
        self._qudit_to_index_map = {
            qubit: i for i, qubit in enumerate(sorted(self.all_qubits()))
        }
        print("Qudit to index map:", self._qudit_to_index_map)
        self._mps_operations = self._translate_to_mps_operations()

    # def _resolve_parameters_(self, param_resolver: cirq.study.ParamResolver):
    #     """Returns a circuit with all parameters resolved by the param_resolver.
    #
    #     Args:
    #         param_resolver: Defines values for parameters in the circuit.
    #     """
    #     mpsim_circuit = super()._resolve_parameters_(param_resolver)
    #     mpsim_circuit.device = self.device
    #     return mpsim_circuit

    # TODO: Should this keep the same moment/operation circuit structure?
    #  Or should it just be one-dimensional?
    def _translate_to_mps_operations(self) -> List[MPSOperation]:
        """Appends all operations in a circuit to MPS Operations."""
        all_mps_operations = []
        for (moment_index, moment) in enumerate(self):
            for operation in moment:
                all_mps_operations.append(
                    MPSOperation.from_gate_operation(
                        operation,
                        self._qudit_to_index_map
                    )
                )
        return all_mps_operations

    # TODO: Every time a gate is added to the circuit, also add it to
    #  self._mpsim_operations. E.g.,
    #  mpsim_circuit.append([some new gates])
    #    or
    #  mpsim_circuit.insert([some gates at some location])
    #  Should update mpsim_circuit._mpsim_operations
    #  Otherwise, a new MPSimCircuit will need to be created before every
    #  circuit simulation.
    #  In which case it would just be better to let the MPS Simulator
    #  handle the conversion.
