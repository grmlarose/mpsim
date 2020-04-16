"""Unit tests for MPSimulator."""

import numpy as np

import cirq

from mpsim import MPS
from mpsim.mpsim_cirq.circuits import MPSimCircuit
from mpsim.mpsim_cirq.simulator import MPSimulator


def test_simulate_bell_state_cirq_circuit():
    """Tests correctness for the final MPS wavefunction when simulating a
    Cirq Circuit which prepares a Bell state.
    """
    # Define the circuit
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        cirq.ops.H.on(qreg[0]),
        cirq.ops.CNOT(*qreg)
    )

    # Do the simulation using the MPS Simulator
    sim = MPSimulator()
    res = sim.simulate(circ)
    assert isinstance(res, MPS)
    print(res.wavefunction)
    assert np.allclose(
        res.wavefunction, np.array([1., 0., 0., 1.]) / np.sqrt(2)
    )


def test_simulate_bell_state_mpsim_circuit():
    """Tests correctness for the final MPS wavefunction when simulating a
    Cirq Circuit which prepares a Bell state.
    """
    # Define the circuit
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        cirq.ops.H.on(qreg[0]),
        cirq.ops.CNOT(*qreg)
    )

    # Convert to an MPSimCircuit
    mpsim_circ = MPSimCircuit(circ)

    # Do the simulation using the MPS Simulator
    sim = MPSimulator()
    res = sim.simulate(mpsim_circ)
    assert isinstance(res, MPS)
    print(res.wavefunction)
    assert np.allclose(
        res.wavefunction, np.array([1., 0., 0., 1.]) / np.sqrt(2)
    )


def test_simulate_one_dimension_supremacy_circuit():
    """Tests simulating a one-dimensional supremacy circuit
    using the MPSimulator.
    """
    # Get the circuit
    circuit = cirq.experiments.generate_boixo_2018_supremacy_circuits_v2_grid(
        n_rows=1, n_cols=5, cz_depth=10, seed=1
    )
    print(circuit)

    # Do the simulation using the MPS Simulator
    sim = MPSimulator()
    res = sim.simulate(circuit)
    assert isinstance(res, MPS)
    assert np.isclose(res.norm(), 1.)
