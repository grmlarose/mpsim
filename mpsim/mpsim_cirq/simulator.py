"""Defines MPSIM Simulator for Cirq circuits."""

from typing import Any, List, Sequence, Union

from cirq import Circuit, ops, protocols, study
from cirq.sim import (
    SimulatesAmplitudes, SimulatesFinalState, SimulationTrialResult
)

from mpsim import MPS
from mpsim.mpsim_cirq.circuits import (
    MPSimCircuit, mps_operation_from_gate_operation
)


class MPSimulator(SimulatesFinalState):

    def __init__(self, options: dict = {}):
        """Initializes and MPS Simulator.

        Args:
            options: Dictionary of options for the simulator.

            Valid options:
                "maxsvals" (int): Number of singular values to keep after each
                                  two qubit gate.

                "fraction" (float): Number of singular values to keep expressed
                                    as a fraction of the maximum bond dimension
                                    for the given tensor.
        """
        self._options = options

    # def compute_amplitudes_sweep(
    #     self,
    #     program: Union[Circuit, MPSimCircuit],
    #     bitstrings: Sequence[str],
    #     params: study.Sweepable,
    #     qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT
    # ) -> Sequence[Sequence[complex]]:
    #     raise NotImplementedError()

    def simulate_sweep(
            self,
            program: Union[Circuit, MPSimCircuit],
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Any = None,
    ) -> List[Any]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation.  See
                documentation of the implementing class for details.
        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        if not isinstance(program, (Circuit, MPSimCircuit)):
            raise ValueError(
                f"Program is of type {type(program)} but should be either"
                " a cirq.Circuit or mpsim.mpsim_cirq.MPSimCircuit."
            )
        # TODO: This throws an error if any gates are parameterized because
        #  these parameterized gates will not have a _unitary_ method until
        #  they are solved by
        #  param_resolvers = study.to_resolvers(params)
        #  solved_circuit = protocols.resolve_parameters(program, prs)
        # if isinstance(program, Circuit):
        #     program = MPSimCircuit(
        #         program, device=program.device
        #     )

        param_resolvers = study.to_resolvers(params)

        trial_results = []
        for prs in param_resolvers:
            solved_circuit = protocols.resolve_parameters(program, prs)

            ordered_qubits = ops.QubitOrder.as_qubit_order(
                qubit_order).order_for(
                solved_circuit.all_qubits())

            qubit_to_index_map = {
                qubit: index for index, qubit in enumerate(ordered_qubits)
            }

            mps = MPS(nqudits=len(solved_circuit.all_qubits()))
            # TODO: Account for an input ordering of operations to apply here
            for gate_operation in solved_circuit.all_operations():
                mps_operation = mps_operation_from_gate_operation(
                    gate_operation, qubit_to_index_map
                )
                mps.apply_mps_operation(mps_operation, **self._options)
            trial_results.append(mps)
        return trial_results
