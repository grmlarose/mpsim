"""Defines MPSIM circuits as extensions of Cirq circuits."""

import cirq


class MPSimCircuit(cirq.Circuit):
    """Defines MPS Circuits which extend cirq.Circuits and 
    can be simulated by an MPSSimulator.
    """
    def __init__(
        self, 
        cirq_circuit: cirq.Circuit,
        device: cirq.devices = cirq.devices.UNONSTRAINED_DEVICE
    ) -> None:
    """Constructor for MPSimCircuit.
    
    Args:
        cirq_circuit: Cirq circuit to create an MPS Sim circuit from.
        device: Device the circuit runs on.
    """
    super().__init__(cirq_circuit, device=device)

