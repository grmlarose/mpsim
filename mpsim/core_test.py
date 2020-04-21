"""Unit tests for inital MPS states."""

import pytest

import numpy as np
import tensornetwork as tn

from mpsim import MPS

from mpsim.gates import (
    igate,
    xgate,
    zgate,
    hgate,
    cnot,
    cphase,
    zero_state,
    one_state,
    plus_state,
)

from mpsim.mpsim_cirq.circuits import MPSOperation


def test_mps_one_qudit():
    """Ensures an error is raised if the number of qudits is less than two."""
    for d in (2, 3, 10, 20, 100):
        with pytest.raises(ValueError):
            MPS(nqudits=1, qudit_dimension=d)


def test_is_valid_for_product_states():
    """Tests that a product state on different numbers of qudits is valid."""
    for n in range(2, 20):
        for d in range(2, 10):
            mps = MPS(nqudits=n, qudit_dimension=d)
            assert mps.is_valid()


def test_max_bond_dimensions_odd_nqubits():
    """Tests for correctness of maximum bond dimensions for an MPS with
    an odd number of qubits.
    """
    mps = MPS(nqudits=5)
    assert mps._max_bond_dimensions == [2, 4, 4, 2]
    mps = MPS(nqudits=7)
    assert mps._max_bond_dimensions == [2, 4, 8, 8, 4, 2]


def test_max_bond_dimensions_even_nqubits():
    """Tests for correctness of maximum bond dimensions for an MPS
    with an even number of qubits.
    """
    mps = MPS(nqudits=6)
    assert mps._max_bond_dimensions == [2, 4, 8, 4, 2]
    mps = MPS(nqudits=8)
    assert mps._max_bond_dimensions == [2, 4, 8, 16, 8, 4, 2]


def test_max_bond_dimensions_odd_nqudits():
    """Tests for correctness of maximum bond dimensions for an MPS with
    an odd number of qudits.
    """
    d = 4
    mps = MPS(nqudits=5, qudit_dimension=d)
    assert mps._max_bond_dimensions == [4, 16, 16, 4]
    assert mps.wavefunction.shape == (d**5,)

    mps = MPS(nqudits=7, qudit_dimension=d)
    assert mps._max_bond_dimensions == [4, 16, 64, 64, 16, 4]
    assert mps.wavefunction.shape == (d**7,)


def test_max_bond_dimensions_even_nqudits():
    """Tests for correctness of maximum bond dimensions for an MPS
    with an even number of qudits.
    """
    d = 10
    mps = MPS(nqudits=4, qudit_dimension=d)
    assert mps._max_bond_dimensions == [10, 100, 10]
    assert mps.wavefunction.shape == (d ** 4,)

    mps = MPS(nqudits=6, qudit_dimension=d)
    assert mps._max_bond_dimensions == [10, 100, 1000, 100, 10]
    assert mps.wavefunction.shape == (d**6,)


def test_get_max_bond_dimension_qubits():
    """Tests correctness for getting maximum bond dimensions in a qubit MPS."""
    mps = MPS(nqudits=10)
    # Correct max bond dimensions: [2, 4, 8, 16, 32, 16, 8, 4, 2]
    assert mps.get_max_bond_dimension_of(0) == 2
    assert mps.get_max_bond_dimension_of(-1) == 2
    assert mps.get_max_bond_dimension_of(3) == 16
    assert mps.get_max_bond_dimension_of(4) == 32
    assert mps.get_max_bond_dimension_of(5) == 16


def test_get_max_bond_dimension_qudits():
    """Tests correctness for getting maximum bond dimensions in a qudit MPS."""
    d = 10
    mps = MPS(nqudits=6, qudit_dimension=d)
    # Correct max bond dimensions: [10, 100, 1000, 100, 10]
    assert mps.get_max_bond_dimension_of(0) == d
    assert mps.get_max_bond_dimension_of(1) == d**2
    assert mps.get_max_bond_dimension_of(2) == d**3
    assert mps.get_max_bond_dimension_of(3) == d**2
    assert mps.get_max_bond_dimension_of(-1) == d


def test_get_bond_dimensions_product_state():
    """Tests correctness for bond dimensions of a product state MPS."""
    n = 5
    for d in range(3, 10):
        mps = MPS(nqudits=n, qudit_dimension=d)
        assert mps.get_bond_dimensions() == [1] * (n - 1)


def test_get_wavefunction_simple_qubits():
    """Tests getting the wavefunction of a simple qubit MPS."""
    mps = MPS(nqudits=3)
    assert isinstance(mps.wavefunction, np.ndarray)
    assert mps.wavefunction.shape == (8,)
    correct = np.array([1.0] + [0.0] * 7, dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)


def test_get_wavefunction_qutrits_simple():
    """Tests getting the wavefunction of a simple qutrit MPS."""
    mps = MPS(nqudits=3, qudit_dimension=3)
    assert mps.wavefunction.shape == (27,)
    assert np.allclose(mps.wavefunction, [1] + [0] * 26)
    assert mps.is_valid()


def test_get_wavefunction_deosnt_modify_mps_qubits():
    """Tests that getting the wavefunction doesn't affect the nodes of a
    qubit MPS.
    """
    mps = MPS(nqudits=2)
    left_node, right_node = mps.get_nodes(copy=False)
    _ = mps.wavefunction
    assert len(left_node.edges) == 2
    assert len(left_node.get_all_nondangling()) == 1
    assert len(left_node.get_all_dangling()) == 1
    assert len(right_node.edges) == 2
    assert len(right_node.get_all_nondangling()) == 1
    assert len(right_node.get_all_dangling()) == 1


def test_get_wavefunction_deosnt_modify_mps_qudits():
    """Tests that getting the wavefunction doesn't affect the nodes of a
     qudit MPS.
     """
    mps = MPS(nqudits=2, qudit_dimension=5)
    left_node, right_node = mps.get_nodes(copy=False)
    _ = mps.wavefunction
    assert len(left_node.edges) == 2
    assert len(left_node.get_all_nondangling()) == 1
    assert len(left_node.get_all_dangling()) == 1
    assert len(right_node.edges) == 2
    assert len(right_node.get_all_nondangling()) == 1
    assert len(right_node.get_all_dangling()) == 1


def test_correctness_of_initial_product_state_two_qubits():
    """Tests that the contracted MPS is indeed the all zero state
    for two qubits.
    """
    mps = MPS(nqudits=2)
    lq, _ = mps.get_nodes()
    wavefunction_node = tn.contract(lq[1])
    wavefunction = np.reshape(wavefunction_node.tensor, newshape=(4,))
    correct = np.array([1, 0, 0, 0], dtype=np.complex64)
    assert np.allclose(wavefunction, correct)


def test_correctness_of_initial_product_state():
    """Tests that the contracted MPS is indeed the all zero state
    for multiple qudits.
    """
    for n in range(3, 10):
        for d in range(2, 5):
            mps = MPS(n, d)
            wavefunction = mps.wavefunction
            correct = np.array([1] + [0] * (d ** n - 1), dtype=np.complex64)
            assert np.allclose(wavefunction, correct)


# Unit tests for applying gates to qubit (as opposed to qudit) MPS
@pytest.mark.parametrize(
    ["gate", "expected"],
    [(xgate(), one_state),
     (hgate(), plus_state),
     (zgate(), zero_state),
     (igate(), zero_state)],
)
def test_apply_one_qubit_gate(gate, expected):
    """Tests application of a single qubit gate to several MPS."""
    for n in range(2, 8):
        for j in range(n):
            mps = MPS(n)
            mps.apply_one_qubit_gate(gate, j)
            final_state = np.reshape(mps.get_node(j).tensor, newshape=(2,))
            assert np.allclose(final_state, expected)


def test_apply_oneq_gate_to_all():
    """Tests correctness for final wavefunction after applying a
    NOT gate to all qubits in a two-qubit MPS.
    """
    mps = MPS(nqudits=2)
    mps.apply_one_qubit_gate_to_all(xgate())
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)


def test_apply_oneq_gate_to_all_hadamard():
    """Tests correctness for final wavefunction after applying a Hadamard
    gate to all qubits in a five-qubit MPS.
    """
    n = 5
    mps = MPS(nqudits=n)
    mps.apply_one_qubit_gate_to_all(hgate())
    correct = 1 / 2 ** (n / 2) * np.ones(2 ** n)
    assert np.allclose(mps.wavefunction, correct)


def test_apply_twoq_cnot_two_qubits():
    """Tests for correctness of final wavefunction after applying a CNOT
    to a two-qubit MPS.
    """
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.apply_two_qubit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|00> = |00>
    mps = MPS(nqudits=2)
    mps.apply_two_qubit_gate(cnot(), 0, 1)
    correct = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|01> = |01>
    mps = MPS(nqudits=2)
    mps.x(1)
    mps.apply_two_qubit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|11> = |10>
    mps = MPS(nqudits=2)
    mps.x(-1)  # Applies to all qubits in the MPS
    mps.apply_two_qubit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)


def test_apply_twoq_cnot_two_qubits_flipped_control_and_target():
    """Tests for correctness of final wavefunction after applying a CNOT
    to a two-qubit MPS.
    """
    # In the following tests, the first qubit is always the target qubit.
    # Check that CNOT|10> = |10>
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.h(-1)
    mps.cnot(0, 1)
    mps.h(-1)
    correct = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|00> = |00>
    mps = MPS(nqudits=2)
    mps.h(-1)
    mps.cnot(0, 1)
    mps.h(-1)
    correct = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|01> = |11>
    mps = MPS(nqudits=2)
    mps.x(1)
    mps.h(-1)
    mps.cnot(0, 1)
    mps.h(-1)
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)

    # Check that CNOT|11> = |01>
    mps = MPS(nqudits=2)
    mps.x(-1)
    mps.h(-1)
    mps.cnot(0, 1)
    mps.h(-1)
    correct = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction, correct)


def test_apply_twoq_identical_indices_raises_error():
    """Tests that a two-qubit gate application with
    identical indices raises an error.
    """
    mps2q = MPS(nqudits=2)
    mps3q = MPS(nqudits=3)
    mps9q = MPS(nqudits=9)
    with pytest.raises(ValueError):
        for mps in (mps2q, mps3q, mps9q):
            mps.apply_two_qubit_gate(cnot(), 0, 0)
            mps.cnot(1, 1)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_gate_indexB_great_than_indexA_raise_error(left):
    """Tests that a two-qubit gate with indexA > indexB raises an error.

    TODO: This is really due to my inability to find the bug for this case.
      We can get around this by, e.g. for a CNOT, conjugating by Hadamard gates
      to flip control/target.
    """
    mps = MPS(nqudits=10)
    with pytest.raises(ValueError):
        mps.apply_two_qubit_gate(cnot(), 6, 5, keep_left_canonical=left)
        mps.cnot(6, 5, keep_left_canonical=left)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_cnot_four_qubits_interior_qubits(left):
    """Tests with a CNOT on four qubits acting on "interior" qubits."""
    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(1)  # State: |0100>
    mps.cnot(1, 2, keep_left_canonical=left)  # State: |0110>
    correct = np.zeros(shape=(16,))
    correct[6] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=4)  # State: |0000>
    mps.cnot(1, 2, keep_left_canonical=left)  # State: |0000>
    correct = np.zeros(shape=(16,))
    correct[0] = 1.
    assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_cnot_four_qubits_edge_qubits(left):
    """Tests with a CNOT on four qubits acting on "edge" qubits."""
    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(2)  # State: |0010>
    mps.cnot(2, 3, keep_left_canonical=left)  # State: Should be |0011>
    correct = np.zeros(shape=(16,))
    correct[3] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(0)  # State: |1000>
    mps.cnot(0, 1, keep_left_canonical=left)  # State: Should be |1100>
    correct = np.zeros(shape=(16,))
    correct[12] = 1.
    assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_cnot_five_qubits_all_combinations(left):
    """Tests applying a CNOT to a five qubit MPS with all index combinations.

    That is, CNOT_01, CNOT_02, CNOT_03, ..., CNOT_24, CNOT_34 where
    the first number is the control qubit and the second is the target qubit.
    """
    n = 5
    indices = [(a, a + 1) for a in range(n - 1)]

    for (a, b) in indices:
        # Apply the gates
        mps = MPS(n)
        mps.x(a)
        mps.cnot(a, b, keep_left_canonical=left)

        # Get the correct wavefunction
        correct = np.zeros((2 ** n,))
        bits = ["0"] * n
        bits[a] = "1"
        bits[b] = "1"
        correct[int("".join(bits), 2)] = 1.0
        assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_swap_two_qubits(left):
    """Tests swapping two qubits in a two-qubit MPS."""
    mps = MPS(nqudits=2)  # State: |00>
    mps.x(0)  # State: |10>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |00>
    correct = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(1)  # State: |01>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |10>
    correct = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(-1)  # State: |11>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |11>
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_swap_two_qubits(left):
    """Tests swapping two qubits in a two-qubit MPS."""
    mps = MPS(nqudits=2)  # State: |00>
    mps.x(0)  # State: |10>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(1)  # State: |01>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(-1)  # State: |11>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_swap_five_qubits(left):
    """Tests applying a swap gate to an MPS with five qubits."""
    n = 5
    for i in range(n - 1):
        mps = MPS(n)
        mps.x(i)
        mps.swap(i, i + 1, keep_left_canonical=left)
        # Get the correct wavefunction
        correct = np.zeros((2 ** n,))
        bits = ["0"] * n
        bits[i + 1] = "1"
        correct[int("".join(bits), 2)] = 1.0
        assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_qubit_hopping_left_to_right(left):
    """Tests "hopping" a qubit with a sequence of swap gates."""
    n = 8
    mps = MPS(n)
    mps.h(0)
    for i in range(1, n - 1):
        mps.swap(i, i + 1, keep_left_canonical=left)
    correct = np.zeros(2 ** n)
    correct[0] = correct[2 ** (n - 1)] = 1.0 / np.sqrt(2)
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_left_to_right_three_qubits_one_state():
    """Tests moving a node from left to right."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.x(0)                                 # State: |100>
    mps.move_node_from_left_to_right(0, 1)   # State: |010>
    correct = [0., 0., 1., 0., 0., 0., 0., 0.]
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_right_to_left_three_qubits_one_state():
    """Tests moving a node from right to left."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.x(2)                                 # State: |001>
    mps.move_node_from_right_to_left(2, 0)   # State: |100>
    correct = [0., 0., 0., 0., 1., 0., 0., 0.]
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_left_to_right_three_qubits_plus_state():
    """Tests moving a node from left to right."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.h(0)  # State: |000> + |100>
    mps.move_node_from_left_to_right(0, 1)  # State: |000> + |010>
    correct = np.array([1., 0., 1., 0., 0., 0., 0., 0.]) / np.sqrt(2)
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_right_to_left_three_qubits_plus_state():
    """Tests moving a node from right to left."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.h(2)  # State: |000> + |001>
    mps.move_node_from_right_to_left(2, 1)  # State: |000> + |010>
    correct = np.array([1., 0., 1., 0., 0., 0., 0., 0.]) / np.sqrt(2)
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_left_to_right_ten_qubits_end_nodes():
    """Tests moving a node from left to right."""
    n = 10
    mps = MPS(nqudits=n, qudit_dimension=2)  # State: |0000000000>
    mps.x(0)  # State: |1000000000>
    mps.move_node_from_left_to_right(0, 4)  # State: |0000010000>
    correct = np.zeros((2**n,))
    correct[2**5] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps.move_node_from_left_to_right(4, 9)  # State: |0000000001>
    correct = np.zeros((2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_right_to_left_ten_qubits_end_nodes():
    """Tests moving a node from left to right."""
    n = 10
    mps = MPS(nqudits=n, qudit_dimension=2)  # State: |0000000000>
    mps.x(9)  # State: |0000000001>
    mps.move_node_from_right_to_left(9, 5)  # State: |0000010000>
    correct = np.zeros((2**n,))
    correct[2**4] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps.move_node_from_right_to_left(5, 0)  # State: |1000000000>
    correct = np.zeros((2**n,))
    correct[2**(n - 1)] = 1.
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_left_to_right_raises_error_with_left_greater_than_right():
    mps = MPS(nqudits=5)
    with pytest.raises(ValueError):
        mps.move_node_from_left_to_right(
            current_node_index=4, final_node_index=0
        )


def test_move_node_right_to_left_raises_error_with_right_greater_than_left():
    mps = MPS(nqudits=5)
    with pytest.raises(ValueError):
        mps.move_node_from_right_to_left(
            current_node_index=0, final_node_index=4
        )


def test_move_node_left_to_right_then_apply_two_qubit_gate():
    n = 5
    mps = MPS(nqudits=n)  # State: |00000>
    mps.x(0)  # State: |10000>
    correct = np.zeros(shape=(2**n,))
    correct[16] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps.swap(3, 4)  # State: |10000>
    assert np.allclose(mps.wavefunction, correct)

    mps.move_node_from_left_to_right(0, 3)  # State: |00010>
    mps.swap(3, 4)  # State: |00001>

    correct = np.zeros(shape=(2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction, correct)


def test_move_node_right_to_left_then_apply_two_qubit_gate():
    n = 5
    mps = MPS(nqudits=n)  # State: |00000>
    mps.x(n - 1)  # State: |00001>
    correct = np.zeros(shape=(2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction, correct)

    mps.swap(0, 1)  # State: |00001>
    assert np.allclose(mps.wavefunction, correct)

    mps.move_node_from_right_to_left(4, 1)  # State: |01000>
    mps.swap(0, 1)  # State: |10000>

    correct = np.zeros(shape=(2**n,))
    correct[2**(n - 1)] = 1.
    assert np.allclose(mps.wavefunction, correct)


def test_move_right_apply_gate_then_move_left():
    """Tests applying a non-adjacent two-qubit gate by moving nodes around
    by swapping. In particular, tests CNOT between the first and last qubits
    in a 3-10 qubit MPS.
    """
    for n in range(3, 10 + 1):
        mps = MPS(nqudits=n)
        mps.x(0)                                    # State: |100>

        # Do SWAPs to implement a non-local gate
        mps.move_node_from_left_to_right(0, n - 2)  # State: |010>
        mps.cnot(n - 2, n - 1)                      # State: |011>

        # Invert the SWAP network
        mps.move_node_from_right_to_left(n - 2, 0)  # State: |101>
        correct = np.zeros(shape=(2**n,))
        correct[2**(n - 1) + 1] = 1.
        assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_bell_state(left):
    """Tests for wavefunction correctness after preparing a Bell state."""
    n = 2
    mps = MPS(n)
    mps.h(0)
    mps.cnot(0, 1, keep_left_canonical=left)
    correct = 1.0 / np.sqrt(2) * np.array([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_twoq_gates_in_succession(left):
    """Tests for wavefunction correctness after applying a
    series of two-qubit gates.
    """
    n = 2
    mps = MPS(n)
    mps.x(0)  # State: |10>
    mps.h(-1)
    mps.cnot(0, 1, keep_left_canonical=left)
    mps.h(-1)  # State: |10>
    mps.cnot(0, 1, keep_left_canonical=left)  # State: |11>
    mps.x(0)  # State: |01>
    correct = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)


def test_left_vs_right_canonical_two_qubit_one_gate():
    """Performs a two-qubit gate keeping left-canonical and right-canonical,
    checks for equality in final wavefunction.
    """
    n = 2
    lmps = MPS(nqudits=n)
    rmps = MPS(nqudits=n)
    lmps.x(0)
    rmps.x(0)
    lmps.cnot(0, 1)
    rmps.cnot(0, 1)
    lwavefunction = lmps.wavefunction
    rwavefunction = rmps.wavefunction
    cwavefunction = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(lwavefunction, cwavefunction)
    assert np.allclose(rwavefunction, cwavefunction)


def test_apply_cnot_right_to_left_sweep_twoq_mps():
    """Tests applying a CNOT in a "right to left sweep" in a two-qubit MPS."""
    n = 2
    mps = MPS(n)
    mps.x(1)
    mps.h(-1)
    mps.cnot(0, 1, keep_left_canonical=False)
    mps.h(-1)

    mps.h(-1)
    mps.cnot(0, 1, keep_left_canonical=False)
    mps.h(-1)

    mps.cnot(0, 1, keep_left_canonical=False)
    mps.cnot(0, 1, keep_left_canonical=False)
    assert mps.is_valid()


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_valid_mps_indexA_greater_than_indexB_twoq_three_qubits(left):
    """Tests successive application of two CNOTs in a three-qubit MPS."""
    n = 3
    mps = MPS(n)
    mps.x(0)
    mps.cnot(0, 1, keep_left_canonical=left)
    assert mps.is_valid()

    mps.h(0)
    mps.h(1)
    mps.cnot(0, 1, keep_left_canonical=left)
    mps.h(0)
    mps.h(1)
    assert mps.is_valid()


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_three_cnots_is_swap(left):
    for n in range(2, 11):
        mps = MPS(n)
        mps.x(0)

        # CNOT(0, 1)
        mps.cnot(0, 1, keep_left_canonical=left)

        # CNOT(1, 0)
        mps.h(-1)
        mps.cnot(0, 1, keep_left_canonical=left)
        mps.h(-1)

        # CNOT(0, 1)
        mps.cnot(0, 1)

        correct = np.zeros((2 ** n))
        correct[2 ** (n - 2)] = 1
        assert np.allclose(mps.wavefunction, correct)


def test_apply_cnot_right_to_left_sweep_threeq_mps():
    """Tests applying a CNOT in a "right to left sweep" in a
    three-qubit MPS retains a valid MPS.
    """
    n = 3
    mps = MPS(n)
    mps.x(2)
    mps.cnot(1, 2, keep_left_canonical=True)
    mps.cnot(0, 1, keep_left_canonical=True)
    assert mps.is_valid()


def test_qubit_hopping_left_to_right_and_back():
    """Tests "hopping" a qubit with a sequence of swap gates in
     several n-qubit MPS states.
     """
    for n in range(2, 20):
        print("Status: n =", n)
        mps = MPS(n)
        mps.x(0)
        for i in range(n - 1):
            mps.swap(i, i + 1, keep_left_canonical=True)
        for i in range(n - 1, 0, -1):
            mps.swap(i - 1, i, keep_left_canonical=True)
        assert mps.is_valid()
        correct = np.zeros(2 ** n)
        correct[2 ** (n - 1)] = 1
        assert np.allclose(mps.wavefunction, correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_cnot_truncation_two_qubits_product(left):
    """Tests applying a CNOT with truncation on a product state."""
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.cnot(0, 1, max_singular_values=0.5, keep_left_canonical=left)
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction, correct)


def test_cnot_truncation_on_bell_state():
    """Tests CNOT with truncation on the state |00> + |10>."""
    # Test with truncation
    mps = MPS(nqudits=2)
    mps.h(0)
    mps.cnot(0, 1, fraction=0.5)
    correct = np.array([1 / np.sqrt(2), 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction, correct)

    # Test keeping all singular values ==> Bell state
    mps = MPS(nqudits=2)
    mps.h(0)
    mps.cnot(0, 1, fraction=1)
    correct = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)])
    assert np.allclose(mps.wavefunction, correct)


def test_bond_dimension_doubles_two_qubit_gate():
    """Tests that the bond dimension doubles after applying a
    two-qubit gate to a product state.
    """
    mps = MPS(nqudits=2)
    assert mps.get_bond_dimension_of(0) == 1
    mps.h(0)
    assert mps.get_bond_dimension_of(0) == 1
    mps.cnot(0, 1)
    assert mps.is_valid()
    assert mps.get_bond_dimension_of(0) == 2
    mps.cnot(0, 1)
    assert mps.get_bond_dimension_of(0) == 2


def test_keep_half_bond_dimension_singular_values():
    """Tests keeping a number of singular values which is half
    the maximum bond dimension.
    """
    # Get an MPS and test the initial bond dimensions and max bond dimensions
    mps = MPS(nqudits=4)
    assert mps.get_bond_dimensions() == [1, 1, 1]
    assert mps.get_max_bond_dimensions() == [2, 4, 2]
    
    # Apply a two qubit gate explicitly keeping all singular values
    mps.r(-1)
    mps.apply_two_qubit_gate(
        cnot(), 0, 1, fraction=1,
    )
    assert mps.get_bond_dimensions() == [2, 1, 1]
    
    # Get an MPS and test the initial bond dimensions and max bond dimensions
    mps = MPS(nqudits=4)
    assert mps.get_bond_dimensions() == [1, 1, 1]
    assert mps.get_max_bond_dimensions() == [2, 4, 2]
    
    # Apply a two qubit gate keeping half the singular values
    mps.r(-1)
    mps.apply_two_qubit_gate(
        cnot(), 0, 1, fraction=0.5
    )
    assert mps.get_bond_dimensions() == [1, 1, 1]
    
    
def test_norm_two_qubit_product_simple():
    """Tests norm of a two-qubit product state MPS."""
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    
    # Make sure the wavefunction hasn't changed
    assert np.allclose(mps.wavefunction, [1, 0, 0, 0])


@pytest.mark.parametrize(["n"], 
                          [[3], [4], [5], [6], [7], [8], [9], [10]]
                         )
def test_norm_nqubit_product_state(n):
    """Tests n qubit MPS in the all |0> state have norm 1."""
    assert MPS(nqudits=n).norm() == 1


def test_norm_after_local_rotations():
    """Applies local rotations (single qubit gates) to an MPS and ensures
    the norm stays one.
    """
    mps = MPS(nqudits=10)
    assert mps.norm() == 1
    mps.h(-1)
    assert np.isclose(mps.norm(), 1.)


def test_norm_after_two_qubit_gate():
    """Tests computing the norm of an MPS after a two-qubit gate."""
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    mps.h(0)
    mps.cnot(0, 1)
    assert np.isclose(mps.norm(), 1.0)


def test_norm_decreases_after_two_qubit_gate_with_truncation():
    """Tests that the norm of an MPS decreases when we throw away svals."""
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    mps.h(0)
    mps.cnot(0, 1, maxsvals=1)
    assert mps.norm() < 1.0  # TODO: Check it agrees with norm from wavefunction


def test_norm_is_zero_after_throwing_away_all_singular_values():
    """Does a two-qubit gate and throws away all singular values,
    checks that the norm is zero.
    """
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    mps.h(0)
    mps.cnot(0, 1, maxsvals=0)
    assert mps.norm() == 0


def test_apply_one_qubit_mps_operation_xgate():
    """Tests applying a single qubit MPS Operation."""
    mps = MPS(nqudits=2)
    mps_operation = MPSOperation(xgate(), qudit_indices=(0,), qudit_dimension=2)
    assert np.allclose(mps.wavefunction, [1., 0., 0., 0.])

    mps.apply_mps_operation(mps_operation)  # Applies NOT to the first qubit
    assert np.allclose(mps.wavefunction, [0., 0., 1., 0.])


def test_mps_operation_prepare_bell_state():
    """Tests preparing a Bell state using MPS Operations."""
    mps = MPS(nqudits=2)
    h_op = MPSOperation(hgate(), qudit_indices=(0,), qudit_dimension=2)
    cnot_op = MPSOperation(cnot(), qudit_indices=(0, 1), qudit_dimension=2)
    assert np.allclose(mps.wavefunction, [1., 0., 0., 0.])

    mps.apply_mps_operation(h_op)
    mps.apply_mps_operation(cnot_op)
    correct = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
    assert np.allclose(mps.wavefunction, correct)


def test_mps_operation_prepare_bell_state_with_truncation():
    """Tests preparing a Bell state using MPS Operations providng maxsvals
    as a keyword argument to MPS.apply_mps_operation.
    """
    mps = MPS(nqudits=2)
    h_op = MPSOperation(hgate(), qudit_indices=(0,), qudit_dimension=2)
    cnot_op = MPSOperation(cnot(), qudit_indices=(0, 1), qudit_dimension=2)
    assert np.allclose(mps.wavefunction, [1., 0., 0., 0.])

    mps.apply_mps_operation(h_op)
    mps.apply_mps_operation(cnot_op, maxsvals=1)
    correct = 1 / np.sqrt(2) * np.array([1., 0., 0., 0.])
    assert np.allclose(mps.wavefunction, correct)


def test_apply_nonlocal_two_qubit_gate():
    """Tests applying a non-local CNOT in a 3-10 qubit MPS."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)      # State: |00...0>
        mps.x(0)                  # State: |10...0>
        mps.cnot(0, n - 1)        # State: |10...1>
        correct = np.zeros(shape=(2**n,))
        correct[2**(n - 1) + 1] = 1.
        assert np.allclose(mps.wavefunction, correct)


def test_prepare_ghz_states_using_nonlocal_gates():
    """Tests preparing n-qubit GHZ states using non-local CNOT gates."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)
        mps.h(0)
        for i in range(1, n):
            mps.cnot(0, i)
        correct = np.zeros(shape=(2**n,))
        correct[0] = correct[-1] = 1. / np.sqrt(2)
        assert np.allclose(mps.wavefunction, correct)


def test_apply_qft_nonlocal_gates():
    """Tests applying the QFT to an n-qubit MPS in the all zero state."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)
        for i in range(n - 1, -1, -1):
            mps.h(i)
            for j in range(i - 1, 0, -1):
                mps.apply_two_qubit_gate(cphase(2**(i - j)), j, i)
        correct = np.ones(shape=(2**n,))
        correct /= 2**(n / 2)
        assert np.allclose(mps.wavefunction, correct)
