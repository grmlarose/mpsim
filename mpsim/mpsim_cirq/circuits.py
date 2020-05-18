"""Defines mpsim circuits as extensions of Cirq circuits."""

from typing import Dict, List

import numpy as np

import cirq
import tensornetwork as tn
from mpsim.core import MPSOperation, CannotConvertToMPSOperation


def mps_operation_from_gate_operation(
        gate_operation: cirq.GateOperation,
        qudit_to_index_map: Dict[cirq.Qid, int]
) -> MPSOperation:
    """Constructs an MPS Operation from a gate operation.

    Args:
        gate_operation: A valid cirq.GateOperation or any child class.
        qudit_to_index_map: Dictionary to map qubits to MPS indices.

    Raises:
        CannotConvertToMPSOperation
            If the gate operation does not have a _unitary_ method.
    """
    num_qudits = len(gate_operation.qubits)
    qudit_dimension = 2  # TODO: Check if all Cirq ops are qubit ops
    qudit_indices = tuple(
        [qudit_to_index_map[qudit] for qudit in gate_operation.qubits]
    )

    if not gate_operation._has_unitary_():
        raise CannotConvertToMPSOperation(
            f"Cannot convert operation {gate_operation} into an MPS Operation"
            " because the operation does not have a unitary."
        )

    tensor = gate_operation._unitary_()
    tensor = np.reshape(
        tensor, newshape=[qudit_dimension] * 2 * num_qudits
    )
    node = tn.Node(tensor)
    return MPSOperation(node, qudit_indices, qudit_dimension)


MPSOperation.from_gate_operation = mps_operation_from_gate_operation


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
        super().__init__(cirq_circuit, device=device)
        self._qudit_to_index_map = {
            qubit: i for i, qubit in enumerate(sorted(self.all_qubits()))
        }  # TODO: Account for qubit order instead of always using sorted order.
        self._mps_operations = self._translate_to_mps_operations()

    def _resolve_parameters_(self, param_resolver: cirq.study.ParamResolver):
        """Returns a circuit with all parameters resolved by the param_resolver.

        Args:
            param_resolver: Defines values for parameters in the circuit.
        """
        mpsim_circuit = super()._resolve_parameters_(param_resolver)
        mpsim_circuit.device = self.device
        return mpsim_circuit

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
