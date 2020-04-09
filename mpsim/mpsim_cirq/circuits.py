"""Defines mpsim circuits as extensions of Cirq circuits."""

from typing import List, Tuple

import cirq
from tensornetwork import Node


class MPSOperation:
    """Defines an operation for MPS Simulators."""
    def __init__(
            self,
            gate: Node,
            qudit_indices: Tuple[int],
            qudit_dimension: int = 2
    ) -> None:
        """Constructor for an MPS Instruction.

        Args:
            gate: TensorNetwork node object representing a gate to apply.
                   See Notes below.
            qudit_indices: Indices of qubits to apply the gate to.
            qudit_dimension: Dimension of qudit(s) to which the MPS Operation
                              is applied. Default value is 2 (for qubits).

        Notes:
            Conventions for gates and edges.
                TODO: Add explanation on edge conventions.
        """
        self._gate = gate
        self._qudit_indices = qudit_indices
        self._qudit_dimension = qudit_dimension

    @property
    def num_qudits(self) -> int:
        """Returns the number of qubits the MPS Operation acts on."""
        return len(self._qudit_indices)

    def _is_valid(self) -> bool:
        """Returns True if the MPS Operation is valid, else False.

        A valid MPS Operation meets the following criteria:
            (1) Tensor of gate is d x d where d = qudit dimension.
            (2) Tensor has 2n free edges where n = number of qudits.
        """
        pass

    def _is_unitary(self) -> bool:
        """Returns True if the MPS Operation is unitary, else False.

        An MPS Operation is unitary if its gate tensor U is unitary, i.e. if
        U^dag @ U = U @ U^dag = I.
        """
        pass


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
        self._mpsim_instructions = self._translate_to_mpsim_operations()

    # TODO: Should this keep the same moment/operation circuit structure?
    #  Or should it just be one-dimensional?
    def _translate_to_mpsim_operations(self) -> List[MPSOperation]:
        """Appends all operations in a circuit to MPS operations."""
        pass

    # TODO: Every time a gate is added to the circuit, also add it to
    #  self._mpsim_operations. E.g.,
    #  mpsim_circuit.append([some new gates])
    #  or
    #  mpsim_circuit.insert([some gates at some location])
    #  Should update mpsim_circuit._mpsim_operations
