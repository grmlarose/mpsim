"""Unit tests for MPSimulator."""

import numpy as np
import sympy

import cirq

import pytest

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


def test_two_qubit_parameterized_circuit_single_parameter():
    """Tests a two-qubit circuit with a single parameter."""
    theta_name, theta_value = "theta", 1.0
    theta = sympy.Symbol(name=theta_name)
    qreg = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        cirq.Ry(theta).on(qreg[0]),
        cirq.CNOT.on(*qreg)
    )

    sim = MPSimulator()
    mps = sim.simulate(
        circ, param_resolver=cirq.ParamResolver({theta_name: theta_value})
    )
    solved_circuit = cirq.Circuit(
        cirq.Ry(theta_value).on(qreg[0]),
        cirq.CNOT.on(*qreg)
    )
    assert np.allclose(mps.wavefunction, solved_circuit.final_wavefunction())


def test_parameterized_single_qubit_gates():
    """Tests several different single-qubit gates with parameters."""
    rng = np.random.RandomState(seed=1)
    n = 4
    symbols = [sympy.Symbol(str(i)) for i in range(n)]
    qreg = cirq.LineQubit.range(n)

    num_tests = 20
    for _ in range(num_tests):
        values = list(rng.rand(n))
        circ = cirq.Circuit(
            cirq.ops.HPowGate(exponent=symbols[0]).on(qreg[0]),
            cirq.ops.ZPowGate(exponent=symbols[1]).on(qreg[1]),
            cirq.ops.PhasedXPowGate(phase_exponent=symbols[2]).on(qreg[2]),
            cirq.ops.Ry(rads=symbols[3]).on(qreg[3]),
        )

        # Get the final wavefunction using the Cirq Simulator
        solved_circuit = cirq.Circuit(
            cirq.ops.HPowGate(exponent=values[0]).on(qreg[0]),
            cirq.ops.ZPowGate(exponent=values[1]).on(qreg[1]),
            cirq.ops.PhasedXPowGate(phase_exponent=values[2]).on(qreg[2]),
            cirq.ops.Ry(rads=values[3]).on(qreg[3]),
        )
        cirq_wavefunction = solved_circuit.final_wavefunction()

        # Get the final wavefunction using the MPS Simulator
        sim = MPSimulator()
        mps = sim.simulate(circ, dict(zip(symbols, values)))

        assert np.allclose(mps.wavefunction, cirq_wavefunction)


def test_parameterized_local_two_qubit_gates():
    """Tests several different two-qubit local gates with parameters."""
    rng = np.random.RandomState(seed=1)
    n = 4
    symbols = [sympy.Symbol(str(i)) for i in range(n // 2)]
    qreg = cirq.LineQubit.range(n)

    num_tests = 20
    for _ in range(num_tests):
        values = list(rng.rand(n // 2))
        circ = cirq.Circuit(
            cirq.ops.H.on_each(*qreg),
            cirq.ops.CZPowGate(exponent=symbols[0]).on(qreg[0], qreg[1]),
            cirq.ops.ZZPowGate(exponent=symbols[1]).on(qreg[2], qreg[3])
        )

        # Get the final wavefunction using the Cirq Simulator
        solved_circuit = cirq.Circuit(
            cirq.ops.H.on_each(*qreg),
            cirq.ops.CZPowGate(exponent=values[0]).on(qreg[0], qreg[1]),
            cirq.ops.ZZPowGate(exponent=values[1]).on(qreg[2], qreg[3])
        )
        cirq_wavefunction = solved_circuit.final_wavefunction()

        # Get the final wavefunction using the MPS Simulator
        sim = MPSimulator()
        mps = sim.simulate(circ, dict(zip(symbols, values)))

        assert np.allclose(mps.wavefunction, cirq_wavefunction)


def test_parameterized_nonlocal_two_qubit_gates():
    """Tests a non-local two-qubit gate with a parameter."""
    rng = np.random.RandomState(seed=1)
    symbols = [sympy.Symbol("theta")]
    qreg = cirq.LineQubit.range(3)

    num_tests = 20
    for _ in range(num_tests):
        values = list(rng.rand(1))
        circ = cirq.Circuit(
            cirq.ops.H.on(qreg[0]),
            cirq.ops.X.on(qreg[1]),
            cirq.ops.CZPowGate(exponent=symbols[0]).on(qreg[0], qreg[2]),
        )

        # Get the final wavefunction using the Cirq Simulator
        solved_circuit = cirq.Circuit(
            cirq.ops.H.on(qreg[0]),
            cirq.ops.X.on(qreg[1]),
            cirq.ops.CZPowGate(exponent=values[0]).on(qreg[0], qreg[2]),
        )
        cirq_wavefunction = solved_circuit.final_wavefunction()

        # Get the final wavefunction using the MPS Simulator
        sim = MPSimulator()
        mps = sim.simulate(circ, dict(zip(symbols, values)))

        assert np.allclose(mps.wavefunction, cirq_wavefunction)


def test_three_qubit_gate_raise_value_error():
    """Tests that a ValueError is raised when attempting to simulate a circuit
    with a three-qubit gate in it.
    """
    qreg = [cirq.GridQubit(x, 0) for x in range(3)]
    circ = cirq.Circuit(
        cirq.ops.TOFFOLI.on(*qreg)
    )
    with pytest.raises(ValueError):
        MPSimulator().simulate(circ)


@pytest.mark.parametrize("nqubits", [2, 4, 8])
def test_random_circuits(nqubits: int):
    """Tests several random circuits and checks the output wavefunction against
    the Cirq simulator.
    """
    np.random.seed(1)
    for _ in range(50):
        circuit = cirq.testing.random_circuit(
            qubits=nqubits,
            n_moments=20,
            op_density=0.95
        )
        correct = circuit.final_wavefunction()
        mps = MPSimulator().simulate(circuit)
        assert np.allclose(mps.wavefunction, correct)
