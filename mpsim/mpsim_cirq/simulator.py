"""Defines MPSIM Simulator for Cirq circuits."""

from typing import Sequence

import cirq


class MPSSimulator(cirq.SimulatesAmplitudes, cirq.SimulatesFinalState):
    
    def compute_amplitudes_sweep(
        self,
        program: cirq.Circuit,  # Or mpsim.mpsim_cirq.circuits.MPSIMCircuit
        bitstrings: Sequence[str],
        params: cirq.study.Sweepable,
        qubit_order: cirq.ops.QubitOrderOrList = cirq.ops.QubitOrder.Default
    ) -> Sequence[Sequence[complex]]:
        pass
    
    def simulate_sweep():
        pass

