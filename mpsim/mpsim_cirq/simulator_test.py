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
    assert np.allclose(
        res.wavefunction, np.array([1., 0., 0., 1.]) / np.sqrt(2)
    )


def test_simulate_bell_state_cirq_circuit_with_truncation():
    """Tests correctness for the final MPS wavefunction when simulating a
    Cirq Circuit which prepares a Bell state using only one singular value.
    """
    # Define the circuit
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        cirq.ops.H.on(qreg[0]),
        cirq.ops.CNOT(*qreg)
    )

    # Do the simulation using the MPS Simulator
    sim = MPSimulator(options={"maxsvals": 1})
    res = sim.simulate(circ)
    assert isinstance(res, MPS)
    assert np.allclose(
        res.wavefunction, np.array([1., 0., 0., 0.]) / np.sqrt(2)
    )


def test_simulate_one_dimensional_supremacy_circuit():
    """Tests simulating a one-dimensional supremacy circuit
    using the MPSimulator.
    """
    # Get the circuit
    circuit = cirq.experiments.generate_boixo_2018_supremacy_circuits_v2_grid(
        n_rows=1, n_cols=5, cz_depth=10, seed=1
    )

    # Do the simulation using the MPS Simulator
    sim = MPSimulator()
    res = sim.simulate(circuit)
    assert isinstance(res, MPS)
    assert np.isclose(res.norm(), 1.)


def test_simulate_ghz_circuits():
    """Tests simulating GHZ circuits on multiple qubits."""
    for n in range(3, 10):
        qreg = cirq.LineQubit.range(n)
        circ = cirq.Circuit(
            [cirq.ops.H.on(qreg[0])],
            [cirq.ops.CNOT.on(qreg[0], qreg[i]) for i in range(1, n)]
        )
        cirq_wavefunction = circ.final_wavefunction()
        mps_wavefunction = MPSimulator().simulate(circ).wavefunction
        assert np.allclose(mps_wavefunction, cirq_wavefunction)


def test_simulate_qft_circuit():
    """Tests simulating the QFT circuit on multiple qubits."""
    for n in range(3, 10):
        qreg = cirq.LineQubit.range(n)
        circ = cirq.Circuit()

        # Add the gates for the QFT
        for i in range(n - 1, -1, -1):
            circ.append(cirq.ops.H.on(qreg[i]))
            for j in range(i - 1, -1, -1):
                circ.append(
                    cirq.ops.CZPowGate(exponent=2**(j - i)).on(
                        qreg[j], qreg[i]))
        assert len(list(circ.all_operations())) == n * (n + 1) // 2

        # Check correctness
        cirq_wavefunction = circ.final_wavefunction()
        mps_wavefunction = MPSimulator().simulate(circ).wavefunction
        assert np.allclose(mps_wavefunction, cirq_wavefunction)
