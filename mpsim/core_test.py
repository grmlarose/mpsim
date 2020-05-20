"""Unit tests for inital MPS states."""

from copy import copy
import pytest

import numpy as np
import tensornetwork as tn

from mpsim import MPS, MPSOperation
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
    computational_basis_projector
)
from cirq.qis import density_matrix_from_state_vector


def test_single_qubit_identity_mps_operation():
    """Unit tests for a single-qubit identity MPS Operation."""
    node = igate()
    mps_operation = MPSOperation(node, qudit_indices=0, qudit_dimension=2)
    assert mps_operation.qudit_indices == (0,)
    assert mps_operation.qudit_dimension == 2
    assert mps_operation.is_valid()
    assert mps_operation.is_unitary()
    assert mps_operation.is_single_qudit_operation()
    assert not mps_operation.is_two_qudit_operation()


def test_get_node_and_tensor_one_qubit_mps_operation():
    """Tests getting the node of a one-qubit MPS Operation."""
    np.random.seed(1)
    tensor = np.random.randn(2, 2)
    node = tn.Node(tensor)
    mps_operation = MPSOperation(node, qudit_indices=(0,), qudit_dimension=2)
    copy_node = mps_operation.node(copy=True)
    # TODO: How to check Node equality with tensornetwork?
    assert len(node.edges) == len(copy_node.edges)
    # assert node == copy_node
    copy_tensor = mps_operation.tensor()
    assert np.allclose(tensor, copy_tensor)


def test_two_qubit_mps_operation_cnot():
    """Performs simple checks on a two-qubit CNOT MPS Operation."""
    node = cnot()
    mps_operation = MPSOperation(node, qudit_indices=(0, 1), qudit_dimension=2)
    assert mps_operation.qudit_indices == (0, 1)
    assert mps_operation.qudit_dimension == 2
    assert not mps_operation.is_single_qudit_operation()
    assert mps_operation.is_two_qudit_operation()


def test_two_qubit_mps_operation_nonlocal_cnot():
    """Performs simple checks on a two-qubit non-local CNOT MPS Operation."""
    node = cnot()
    mps_operation = MPSOperation(node, qudit_indices=(0, 2), qudit_dimension=2)
    assert mps_operation.qudit_indices == (0, 2)
    assert mps_operation.is_valid()
    assert not mps_operation.is_single_qudit_operation()
    assert mps_operation.is_two_qudit_operation()


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
    assert mps.wavefunction().shape == (d**5,)

    mps = MPS(nqudits=7, qudit_dimension=d)
    assert mps._max_bond_dimensions == [4, 16, 64, 64, 16, 4]
    assert mps.wavefunction().shape == (d**7,)


def test_max_bond_dimensions_even_nqudits():
    """Tests for correctness of maximum bond dimensions for an MPS
    with an even number of qudits.
    """
    d = 10
    mps = MPS(nqudits=4, qudit_dimension=d)
    assert mps._max_bond_dimensions == [10, 100, 10]
    assert mps.wavefunction().shape == (d ** 4,)

    mps = MPS(nqudits=6, qudit_dimension=d)
    assert mps._max_bond_dimensions == [10, 100, 1000, 100, 10]
    assert mps.wavefunction().shape == (d**6,)


def test_get_max_bond_dimension_qubits():
    """Tests correctness for getting maximum bond dimensions in a qubit MPS."""
    mps = MPS(nqudits=10)
    # Correct max bond dimensions: [2, 4, 8, 16, 32, 16, 8, 4, 2]
    assert mps.max_bond_dimension_of(0) == 2
    assert mps.max_bond_dimension_of(-1) == 2
    assert mps.max_bond_dimension_of(3) == 16
    assert mps.max_bond_dimension_of(4) == 32
    assert mps.max_bond_dimension_of(5) == 16


def test_get_max_bond_dimension_qudits():
    """Tests correctness for getting maximum bond dimensions in a qudit MPS."""
    d = 10
    mps = MPS(nqudits=6, qudit_dimension=d)
    # Correct max bond dimensions: [10, 100, 1000, 100, 10]
    assert mps.max_bond_dimension_of(0) == d
    assert mps.max_bond_dimension_of(1) == d ** 2
    assert mps.max_bond_dimension_of(2) == d ** 3
    assert mps.max_bond_dimension_of(3) == d ** 2
    assert mps.max_bond_dimension_of(-1) == d


def test_get_bond_dimensions_product_state():
    """Tests correctness for bond dimensions of a product state MPS."""
    n = 5
    for d in range(3, 10):
        mps = MPS(nqudits=n, qudit_dimension=d)
        assert mps.bond_dimensions() == [1] * (n - 1)


def test_get_free_edge_of():
    """Tests getting the free edge of nodes in an MPS."""
    for n in range(2, 10):
        for d in (2, 3, 4):
            mps = MPS(nqudits=n, qudit_dimension=d)
            for i in range(n):
                free_edge = mps.get_free_edge_of(i, copy=False)
                assert free_edge.is_dangling()
                assert free_edge.node1.name == f"q{i}"


def test_get_left_connected_edge():
    """Tests getting the left connected edge of nodes in an MPS."""
    for d in (2, 3, 4):
        mps = MPS(nqudits=3, qudit_dimension=d)

        # Left edge of first node should be None
        edge = mps.get_left_connected_edge_of(0)
        assert edge is None

        # Left edge of second node
        edge = mps.get_left_connected_edge_of(1)
        assert not edge.is_dangling()
        assert edge.node1.name == "q0"
        assert edge.node2.name == "q1"

        # Left edge of third node
        edge = mps.get_left_connected_edge_of(2)
        assert not edge.is_dangling()
        assert edge.node1.name == "q2"
        assert edge.node2.name == "q1"


def test_get_right_connected_edge():
    """Tests getting the left connected edge of nodes in an MPS."""
    for d in (2, 3, 4):
        mps = MPS(nqudits=3, qudit_dimension=d)

        # Right edge of first node
        edge = mps.get_right_connected_edge_of(0)
        assert not edge.is_dangling()
        assert edge.node1.name == "q0"
        assert edge.node2.name == "q1"

        # Right edge of second node
        edge = mps.get_right_connected_edge_of(1)
        assert not edge.is_dangling()
        assert edge.node1.name == "q2"
        assert edge.node2.name == "q1"

        # Right edge of third node should be None
        edge = mps.get_right_connected_edge_of(2)
        assert edge is None


def test_get_left_and_get_right_connected_edges():
    """Tests correctness of getting left edges and getting right edges."""
    n = 10
    for d in (2, 3, 4):
        mps = MPS(nqudits=n, qudit_dimension=d)
        for i in range(1, n - 1):
            assert (mps.get_right_connected_edge_of(i - 1) ==
                    mps.get_left_connected_edge_of(i))


def test_from_wavefunction_two_qubits_all_zero_state():
    """Tests constructing an MPS from an initial wavefunction."""
    wavefunction = np.array([1, 0, 0, 0])
    mps = MPS.from_wavefunction(wavefunction, nqudits=2, qudit_dimension=2)
    assert isinstance(mps, MPS)
    assert mps.nqudits == 2
    assert mps.qudit_dimension == 2
    assert np.allclose(mps.wavefunction(), wavefunction)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1.)


def test_from_wavefunction_three_qubits_all_zero_state():
    """Tests constructing an MPS from an initial wavefunction."""
    wavefunction = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    mps = MPS.from_wavefunction(wavefunction, nqudits=3, qudit_dimension=2)
    assert isinstance(mps, MPS)
    assert mps.nqudits == 3
    assert mps.qudit_dimension == 2
    assert np.allclose(mps.wavefunction(), wavefunction)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1.)


def test_from_wavefunction_random_qubit_wavefunctions():
    """Tests correctness of MPS wavefunction for creating an MPS from
    several initial random wavefunctions.
    """
    np.random.seed(1)
    for _ in range(100):
        for n in range(2, 8):
            wavefunction = np.random.rand(2**n)
            wavefunction /= np.linalg.norm(wavefunction, ord=2)
            mps = MPS.from_wavefunction(wavefunction, nqudits=n)
            assert np.allclose(mps.wavefunction(), wavefunction)


def test_from_wavefunction_random_qudit_wavefunctions():
    """Tests correctness of MPS wavefunction for creating an MPS from
    several initial random qudit wavefunctions.
    """
    np.random.seed(11)
    for _ in range(100):
        for n in range(2, 5):
            for d in (2, 3, 4):
                wavefunction = np.random.rand(d**n)
                wavefunction /= np.linalg.norm(wavefunction, ord=2)
                mps = MPS.from_wavefunction(
                    wavefunction, nqudits=n, qudit_dimension=d
                )
                assert np.allclose(mps.wavefunction(), wavefunction)


def test_from_wavefunction_invalid_args():
    """Tests MPS.from_wavefunction raises errors with invalid args."""
    with pytest.raises(TypeError):
        MPS.from_wavefunction({1, 2, 3, 4}, nqudits=2, qudit_dimension=2)

    with pytest.raises(ValueError):
        MPS.from_wavefunction([1., 0., 0., 0.], nqudits=3, qudit_dimension=2)

    with pytest.raises(ValueError):
        MPS.from_wavefunction([1., 0.], nqudits=1, qudit_dimension=2)

    twod_wavefunction = np.array([[1., 0.], [0., 1.]])
    with pytest.raises(ValueError):
        MPS.from_wavefunction(twod_wavefunction, nqudits=2, qudit_dimension=2)


def test_get_wavefunction_simple_qubits():
    """Tests getting the wavefunction of a simple qubit MPS."""
    mps = MPS(nqudits=3)
    assert isinstance(mps.wavefunction(), np.ndarray)
    assert mps.wavefunction().shape == (8,)
    correct = np.array([1.0] + [0.0] * 7, dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)


def test_get_wavefunction_qutrits_simple():
    """Tests getting the wavefunction of a simple qutrit MPS."""
    mps = MPS(nqudits=3, qudit_dimension=3)
    assert mps.wavefunction().shape == (27,)
    assert np.allclose(mps.wavefunction(), [1] + [0] * 26)
    assert mps.is_valid()


def test_get_wavefunction_deosnt_modify_mps_qubits():
    """Tests that getting the wavefunction doesn't affect the nodes of a
    qubit MPS.
    """
    mps = MPS(nqudits=2)
    left_node, right_node = mps.get_nodes(copy=False)
    _ = mps.wavefunction()
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
    _ = mps.wavefunction()
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
            wavefunction = mps.wavefunction()
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
            mps.apply_one_qudit_gate(gate, j)
            final_state = np.reshape(mps.get_node(j).tensor, newshape=(2,))
            assert np.allclose(final_state, expected)


def test_apply_oneq_gate_to_all():
    """Tests correctness for final wavefunction after applying a
    NOT gate to all qubits in a two-qubit MPS.
    """
    mps = MPS(nqudits=2)
    mps.apply_one_qudit_gate_to_all(xgate())
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_oneq_gate_to_all_hadamard():
    """Tests correctness for final wavefunction after applying a Hadamard
    gate to all qubits in a five-qubit MPS.
    """
    n = 5
    mps = MPS(nqudits=n)
    mps.apply_one_qudit_gate_to_all(hgate())
    correct = 1 / 2 ** (n / 2) * np.ones(2 ** n)
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_twoq_cnot_two_qubits():
    """Tests for correctness of final wavefunction after applying a CNOT
    to a two-qubit MPS.
    """
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.apply_two_qudit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|00> = |00>
    mps = MPS(nqudits=2)
    mps.apply_two_qudit_gate(cnot(), 0, 1)
    correct = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|01> = |01>
    mps = MPS(nqudits=2)
    mps.x(1)
    mps.apply_two_qudit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|11> = |10>
    mps = MPS(nqudits=2)
    mps.x(-1)  # Applies to all qubits in the MPS
    mps.apply_two_qudit_gate(cnot(), 0, 1)
    correct = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_twoq_cnot_two_qubits_flipped_control_and_target():
    """Tests for correctness of final wavefunction after applying a CNOT
    to a two-qubit MPS.
    """
    # In the following tests, the first qubit is always the target qubit.
    # Check that CNOT|10> = |10>
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.cnot(1, 0)
    correct = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|00> = |00>
    mps = MPS(nqudits=2)
    mps.cnot(1, 0)
    correct = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|01> = |11>
    mps = MPS(nqudits=2)
    mps.x(1)
    mps.cnot(1, 0)
    correct = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)

    # Check that CNOT|11> = |01>
    mps = MPS(nqudits=2)
    mps.x(-1)
    mps.cnot(1, 0)
    correct = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64)
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_twoq_identical_indices_raises_error():
    """Tests that a two-qubit gate application with
    identical indices raises an error.
    """
    mps2q = MPS(nqudits=2)
    mps3q = MPS(nqudits=3)
    mps9q = MPS(nqudits=9)
    with pytest.raises(ValueError):
        for mps in (mps2q, mps3q, mps9q):
            mps.apply_two_qudit_gate(cnot(), 0, 0)
            mps.cnot(1, 1)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_cnot_four_qubits_interior_qubits(left):
    """Tests with a CNOT on four qubits acting on "interior" qubits."""
    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(1)  # State: |0100>
    mps.cnot(1, 2, keep_left_canonical=left)  # State: |0110>
    correct = np.zeros(shape=(16,))
    correct[6] = 1.
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=4)  # State: |0000>
    mps.cnot(1, 2, keep_left_canonical=left)  # State: |0000>
    correct = np.zeros(shape=(16,))
    correct[0] = 1.
    assert np.allclose(mps.wavefunction(), correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_cnot_four_qubits_edge_qubits(left):
    """Tests with a CNOT on four qubits acting on "edge" qubits."""
    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(2)  # State: |0010>
    mps.cnot(2, 3, keep_left_canonical=left)  # State: Should be |0011>
    correct = np.zeros(shape=(16,))
    correct[3] = 1.
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=4)  # State: |0000>
    mps.x(0)  # State: |1000>
    mps.cnot(0, 1, keep_left_canonical=left)  # State: Should be |1100>
    correct = np.zeros(shape=(16,))
    correct[12] = 1.
    assert np.allclose(mps.wavefunction(), correct)


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
        assert np.allclose(mps.wavefunction(), correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_swap_two_qubits(left):
    """Tests swapping two qubits in a two-qubit MPS."""
    mps = MPS(nqudits=2)  # State: |00>
    mps.x(0)  # State: |10>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |00>
    correct = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(1)  # State: |01>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |10>
    correct = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(-1)  # State: |11>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |11>
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction(), correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_apply_twoq_swap_two_qubits(left):
    """Tests swapping two qubits in a two-qubit MPS."""
    mps = MPS(nqudits=2)  # State: |00>
    mps.x(0)  # State: |10>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(1)  # State: |01>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    mps = MPS(nqudits=2)  # State: |00>
    mps.x(-1)  # State: |11>
    mps.swap(0, 1, keep_left_canonical=left)  # State: |01>
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction(), correct)


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
        assert np.allclose(mps.wavefunction(), correct)


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
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_left_to_right_three_qubits_one_state():
    """Tests moving a node from left to right."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.x(0)                                 # State: |100>
    mps.move_node_from_left_to_right(0, 1)   # State: |010>
    correct = [0., 0., 1., 0., 0., 0., 0., 0.]
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_right_to_left_three_qubits_one_state():
    """Tests moving a node from right to left."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.x(2)                                 # State: |001>
    mps.move_node_from_right_to_left(2, 0)   # State: |100>
    correct = [0., 0., 0., 0., 1., 0., 0., 0.]
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_left_to_right_three_qubits_plus_state():
    """Tests moving a node from left to right."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.h(0)  # State: |000> + |100>
    mps.move_node_from_left_to_right(0, 1)  # State: |000> + |010>
    correct = np.array([1., 0., 1., 0., 0., 0., 0., 0.]) / np.sqrt(2)
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_right_to_left_three_qubits_plus_state():
    """Tests moving a node from right to left."""
    mps = MPS(nqudits=3, qudit_dimension=2)  # State: |000>
    mps.h(2)  # State: |000> + |001>
    mps.move_node_from_right_to_left(2, 1)  # State: |000> + |010>
    correct = np.array([1., 0., 1., 0., 0., 0., 0., 0.]) / np.sqrt(2)
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_left_to_right_ten_qubits_end_nodes():
    """Tests moving a node from left to right."""
    n = 10
    mps = MPS(nqudits=n, qudit_dimension=2)  # State: |0000000000>
    mps.x(0)  # State: |1000000000>
    mps.move_node_from_left_to_right(0, 4)  # State: |0000010000>
    correct = np.zeros((2**n,))
    correct[2**5] = 1.
    assert np.allclose(mps.wavefunction(), correct)

    mps.move_node_from_left_to_right(4, 9)  # State: |0000000001>
    correct = np.zeros((2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_right_to_left_ten_qubits_end_nodes():
    """Tests moving a node from left to right."""
    n = 10
    mps = MPS(nqudits=n, qudit_dimension=2)  # State: |0000000000>
    mps.x(9)  # State: |0000000001>
    mps.move_node_from_right_to_left(9, 5)  # State: |0000010000>
    correct = np.zeros((2**n,))
    correct[2**4] = 1.
    assert np.allclose(mps.wavefunction(), correct)

    mps.move_node_from_right_to_left(5, 0)  # State: |1000000000>
    correct = np.zeros((2**n,))
    correct[2**(n - 1)] = 1.
    assert np.allclose(mps.wavefunction(), correct)


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
    assert np.allclose(mps.wavefunction(), correct)

    mps.swap(3, 4)  # State: |10000>
    assert np.allclose(mps.wavefunction(), correct)

    mps.move_node_from_left_to_right(0, 3)  # State: |00010>
    mps.swap(3, 4)  # State: |00001>

    correct = np.zeros(shape=(2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction(), correct)


def test_move_node_right_to_left_then_apply_two_qubit_gate():
    n = 5
    mps = MPS(nqudits=n)  # State: |00000>
    mps.x(n - 1)  # State: |00001>
    correct = np.zeros(shape=(2**n,))
    correct[1] = 1.
    assert np.allclose(mps.wavefunction(), correct)

    mps.swap(0, 1)  # State: |00001>
    assert np.allclose(mps.wavefunction(), correct)

    mps.move_node_from_right_to_left(4, 1)  # State: |01000>
    mps.swap(0, 1)  # State: |10000>

    correct = np.zeros(shape=(2**n,))
    correct[2**(n - 1)] = 1.
    assert np.allclose(mps.wavefunction(), correct)


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
        assert np.allclose(mps.wavefunction(), correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_bell_state(left):
    """Tests for wavefunction correctness after preparing a Bell state."""
    n = 2
    mps = MPS(n)
    mps.h(0)
    mps.cnot(0, 1, keep_left_canonical=left)
    correct = 1.0 / np.sqrt(2) * np.array([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction(), correct)


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
    assert np.allclose(mps.wavefunction(), correct)


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
    lwavefunction = lmps.wavefunction()
    rwavefunction = rmps.wavefunction()
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
        assert np.allclose(mps.wavefunction(), correct)


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
        mps = MPS(n)
        mps.x(0)
        for i in range(n - 1):
            mps.swap(i, i + 1, keep_left_canonical=True)
        for i in range(n - 1, 0, -1):
            mps.swap(i - 1, i, keep_left_canonical=True)
        assert mps.is_valid()
        correct = np.zeros(2 ** n)
        correct[2 ** (n - 1)] = 1
        assert np.allclose(mps.wavefunction(), correct)


@pytest.mark.parametrize(["left"], [[True], [False]])
def test_cnot_truncation_two_qubits_product(left):
    """Tests applying a CNOT with truncation on a product state."""
    mps = MPS(nqudits=2)
    mps.x(0)
    mps.cnot(0, 1, max_singular_values=0.5, keep_left_canonical=left)
    correct = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mps.wavefunction(), correct)


def test_cnot_truncation_on_bell_state():
    """Tests CNOT with truncation on the state |00> + |10>."""
    # Test with truncation
    mps = MPS(nqudits=2)
    mps.h(0)
    mps.cnot(0, 1, fraction=0.5)
    correct = np.array([1 / np.sqrt(2), 0.0, 0.0, 0.0])
    assert np.allclose(mps.wavefunction(), correct)

    # Test keeping all singular values ==> Bell state
    mps = MPS(nqudits=2)
    mps.h(0)
    mps.cnot(0, 1, fraction=1)
    correct = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)])
    assert np.allclose(mps.wavefunction(), correct)


def test_bond_dimension_doubles_two_qubit_gate():
    """Tests that the bond dimension doubles after applying a
    two-qubit gate to a product state.
    """
    mps = MPS(nqudits=2)
    assert mps.bond_dimension_of(0) == 1
    mps.h(0)
    assert mps.bond_dimension_of(0) == 1
    mps.cnot(0, 1)
    assert mps.is_valid()
    assert mps.bond_dimension_of(0) == 2
    mps.cnot(0, 1)
    assert mps.bond_dimension_of(0) == 2


def test_keep_half_bond_dimension_singular_values():
    """Tests keeping a number of singular values which is half
    the maximum bond dimension.
    """
    # Get an MPS and test the initial bond dimensions and max bond dimensions
    mps = MPS(nqudits=4)
    assert mps.bond_dimensions() == [1, 1, 1]
    assert mps.max_bond_dimensions() == [2, 4, 2]
    
    # Apply a two qubit gate explicitly keeping all singular values
    mps.r(-1)
    mps.apply_two_qudit_gate(
        cnot(), 0, 1, fraction=1,
    )
    assert mps.bond_dimensions() == [2, 1, 1]
    
    # Get an MPS and test the initial bond dimensions and max bond dimensions
    mps = MPS(nqudits=4)
    assert mps.bond_dimensions() == [1, 1, 1]
    assert mps.max_bond_dimensions() == [2, 4, 2]
    
    # Apply a two qubit gate keeping half the singular values
    mps.r(-1)
    mps.apply_two_qudit_gate(
        cnot(), 0, 1, fraction=0.5
    )
    assert mps.bond_dimensions() == [1, 1, 1]


def test_inner_product_basis_states():
    """Tests inner products of four two-qubit basis states."""
    # Get the four MPS
    mps00 = MPS(nqudits=2)
    mps01 = MPS(nqudits=2)
    mps01.apply(MPSOperation(xgate(), 1))
    mps10 = MPS(nqudits=2)
    mps10.apply(MPSOperation(xgate(), 0))
    mps11 = MPS(nqudits=2)
    mps11.apply(MPSOperation(xgate(), 0))
    mps11.apply(MPSOperation(xgate(), 1))
    allmps = (mps00, mps01, mps10, mps11)

    # Test inner products
    for i in range(4):
        for j in range(4):
            assert np.isclose(allmps[i].inner_product(allmps[j]), i == j)


@pytest.mark.parametrize("n", [2, 3, 5, 8, 10])
def test_inner_product_correctness_with_qubit_wavefunctions(n: int):
    """Tests correctness of MPS.inner_product by computing the inner product
    from the wavefunctions.
    """
    np.random.seed(1)

    for _ in range(50):
        # Get the wavefunctions
        wavefunction1 = np.random.randn(2**n) + np.random.randn(2**n) * 1j
        wavefunction1 /= np.linalg.norm(wavefunction1)
        wavefunction2 = np.random.randn(2**n) + np.random.randn(2**n) * 1j
        wavefunction2 /= np.linalg.norm(wavefunction2)

        # Get the MPS from the wavefunctions
        mps1 = MPS.from_wavefunction(
            wavefunction1, nqudits=n, qudit_dimension=2
        )
        mps2 = MPS.from_wavefunction(
            wavefunction2, nqudits=n, qudit_dimension=2
        )

        # Check correctness for the inner products
        assert np.isclose(
            mps1.inner_product(mps2),
            np.inner(wavefunction1, wavefunction2.conj())
        )
        assert np.isclose(
            mps2.inner_product(mps1),
            np.inner(wavefunction2, wavefunction1.conj())
        )


def test_inner_product_raises_error_mismatch_nqudits():
    """Tests that <self|other> raises an error when
    self.nqudits != other.nqudits.
    """
    mps1 = MPS(nqudits=5)
    mps2 = MPS(nqudits=6)
    with pytest.raises(ValueError):
        mps1.inner_product(mps2)


def test_inner_product_raises_error_mismatch_qudit_dimension():
    """Tests that <self|other> raises an error when
    self.qudit_dimension != other.qudit_dimension.
    """
    mps1 = MPS(nqudits=5, qudit_dimension=2)
    mps2 = MPS(nqudits=5, qudit_dimension=3)
    with pytest.raises(ValueError):
        mps1.inner_product(mps2)

    
def test_norm_two_qubit_product_simple():
    """Tests norm of a two-qubit product state MPS."""
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    
    # Make sure the wavefunction hasn't changed
    assert np.allclose(mps.wavefunction(), [1, 0, 0, 0])


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
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))


def test_norm_is_zero_after_throwing_away_all_singular_values():
    """Does a two-qubit gate and throws away all singular values,
    checks that the norm is zero.
    """
    mps = MPS(nqudits=2)
    assert mps.norm() == 1
    mps.h(0)
    mps.cnot(0, 1, maxsvals=0)
    assert mps.norm() == 0


def test_renormalize_mps_which_are_normalized():
    """Makes sure renormalizing a normalized MPS does nothing."""
    for n in range(2, 8):
        for d in (2, 3, 4, 5):
            # All zero state
            mps = MPS(nqudits=n, qudit_dimension=d)
            assert np.isclose(mps.norm(), 1.0)
            mps.renormalize()
            assert np.isclose(mps.norm(), 1.0)

            # Plus state on qubits
            if d == 2:
                ops = [MPSOperation(hgate(), (i,)) for i in range(n)]
                mps.apply(ops)
                assert np.isclose(mps.norm(), 1.0)
                mps.renormalize()
                assert np.isclose(mps.norm(), 1.0)


def test_renormalize_after_throwing_away_singular_values_bell_state():
    """Prepares a Bell state only keeping one singular value with the CNOT
    and checks that renormalization works correctly.
    """
    mps = MPS(nqudits=2)
    mps.apply(
        [MPSOperation(hgate(), (0,)), MPSOperation(cnot(), (0, 1))],
        maxsvals=1
    )
    correct = np.array([1. / np.sqrt(2), 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))

    # Renormalize
    mps.renormalize()
    print(mps.wavefunction())
    correct = np.array([1., 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)
    assert np.isclose(mps.norm(), 1.)


def test_renormalize_to_value_after_throwing_away_singular_values_bell_state():
    """Prepares a Bell state only keeping one singular value with the CNOT
    and checks that renormalization to a provided value works correctly.
    """
    mps = MPS(nqudits=2)
    mps.apply(
        [MPSOperation(hgate(), (0,)), MPSOperation(cnot(), (0, 1))],
        maxsvals=1
    )
    correct = np.array([1. / np.sqrt(2), 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))

    # Renormalize to different values
    for norm in np.linspace(0.1, 2., 100):
        mps.renormalize(to_norm=norm)
        correct = np.array([norm, 0., 0., 0.])
        assert np.allclose(mps.wavefunction(), correct)
        assert np.isclose(mps.norm(), norm)


def test_renormalize_an_mps_with_too_small_norm_raises_error():
    """Asserts that renormalizing an MPS with zero norm raises an error."""
    mps = MPS(nqudits=2)
    mps.apply(
        [MPSOperation(hgate(), (0,)), MPSOperation(cnot(), (0, 1))],
        maxsvals=0
    )
    assert np.isclose(mps.norm(), 0.)
    with pytest.raises(ValueError):
        mps.renormalize()


def test_renormalize_to_invalid_norms_raises_errors():
    """Asserts that renormalizing an MPS to invalid norms raise errors."""
    mps = MPS(nqudits=2)
    with pytest.raises(ValueError):
        mps.renormalize(to_norm=-3.14)

    with pytest.raises(ValueError):
        mps.renormalize(to_norm=1e-20)


def test_apply_one_qubit_mps_operation_xgate():
    """Tests applying a single qubit MPS Operation."""
    mps = MPS(nqudits=2)
    mps_operation = MPSOperation(xgate(), qudit_indices=(0,), qudit_dimension=2)
    assert np.allclose(mps.wavefunction(), [1., 0., 0., 0.])

    mps.apply(mps_operation)  # Applies NOT to the first qubit
    assert np.allclose(mps.wavefunction(), [0., 0., 1., 0.])


def test_mps_operation_prepare_bell_state():
    """Tests preparing a Bell state using MPS Operations."""
    mps = MPS(nqudits=2)
    h_op = MPSOperation(hgate(), qudit_indices=(0,), qudit_dimension=2)
    cnot_op = MPSOperation(cnot(), qudit_indices=(0, 1), qudit_dimension=2)
    assert np.allclose(mps.wavefunction(), [1., 0., 0., 0.])

    mps.apply(h_op)
    mps.apply(cnot_op)
    correct = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
    assert np.allclose(mps.wavefunction(), correct)


def test_mps_operation_prepare_bell_state_with_truncation():
    """Tests preparing a Bell state using MPS Operations providng maxsvals
    as a keyword argument to MPS.apply.
    """
    mps = MPS(nqudits=2)
    h_op = MPSOperation(hgate(), qudit_indices=(0,), qudit_dimension=2)
    cnot_op = MPSOperation(cnot(), qudit_indices=(0, 1), qudit_dimension=2)
    assert np.allclose(mps.wavefunction(), [1., 0., 0., 0.])

    mps.apply(h_op)
    mps.apply(cnot_op, maxsvals=1)
    correct = 1 / np.sqrt(2) * np.array([1., 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_nonlocal_two_qubit_gate():
    """Tests applying a non-local CNOT in a 3-10 qubit MPS."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)      # State: |00...0>
        mps.x(0)                  # State: |10...0>
        mps.cnot(0, n - 1)        # State: |10...1>
        correct = np.zeros(shape=(2**n,))
        correct[2**(n - 1) + 1] = 1.
        assert np.allclose(mps.wavefunction(), correct)


def test_prepare_ghz_states_using_nonlocal_gates():
    """Tests preparing n-qubit GHZ states using non-local CNOT gates."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)
        mps.h(0)
        for i in range(1, n):
            mps.cnot(0, i)
        correct = np.zeros(shape=(2**n,))
        correct[0] = correct[-1] = 1. / np.sqrt(2)
        assert np.allclose(mps.wavefunction(), correct)


def test_apply_qft_nonlocal_gates():
    """Tests applying the QFT to an n-qubit MPS in the all zero state."""
    for n in range(3, 10):
        mps = MPS(nqudits=n)
        for i in range(n - 1, -1, -1):
            mps.h(i)
            for j in range(i - 1, -1, -1):
                mps.apply_two_qudit_gate(cphase(2 ** (j - i)), j, i)
        correct = np.ones(shape=(2**n,))
        correct /= 2**(n / 2)
        assert np.allclose(mps.wavefunction(), correct)


def test_valid_after_orthonormalize_right_edges():
    """Tests |+++> MPS remains valid, retains correct bond dimensions, and
    retains correct wavefunction after orthonormalizing right edges.
    """
    n = 3
    mps = MPS(nqudits=n)
    mps_operations = [MPSOperation(hgate(), (i,)) for i in range(n)]
    mps.apply(mps_operations)
    wavefunction_before = mps.wavefunction()
    assert mps.bond_dimension_of(0) == 1
    assert mps.bond_dimension_of(1) == 1

    # Orthonormalize the right edge of the first node
    mps.orthonormalize_right_edge_of(0)
    assert mps.is_valid()
    assert mps.bond_dimension_of(0) == 1
    assert mps.bond_dimension_of(1) == 1
    assert np.allclose(mps.wavefunction(), wavefunction_before)

    # Orthonormalize the right edge of the second node
    mps.orthonormalize_right_edge_of(1)
    assert mps.is_valid()
    assert mps.bond_dimension_of(0) == 1
    assert mps.bond_dimension_of(1) == 1
    assert np.allclose(mps.wavefunction(), wavefunction_before)


def test_apply_povm_product_state():
    """Tests applying a POVM + orthonormalizing the index to the |+++> state."""
    # Get the projector
    pi0 = computational_basis_projector(state=0)

    # Create an MPS in the H|0> state
    n = 3
    mps = MPS(nqudits=n)  # State: |000>
    mps_operations = [MPSOperation(hgate(), i) for i in range(n)]
    mps.apply(mps_operations)  # State |+++>
    assert np.isclose(mps.norm(), 1.0)
    assert mps.bond_dimensions() == [1, 1]

    # Apply |0><0| to the first qubit
    mps.apply_one_qudit_gate(
        pi0,
        0,
        ortho_after_non_unitary=False,
        renormalize_after_non_unitary=False
    )  # State: 1 / sqrt(2) * |0++>
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))
    assert mps.bond_dimensions() == [1, 1]
    correct = 1. / np.sqrt(2)**3 * np.array([1] * 4 + [0] * 4)
    assert np.allclose(mps.wavefunction(), correct)

    # Apply |0><0| to the second qubit
    mps.apply_one_qudit_gate(
        pi0,
        1,
        ortho_after_non_unitary=False,
        renormalize_after_non_unitary=False
    )  # State: 1 / 2 * |00+>
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / 2.)
    assert mps.bond_dimensions() == [1, 1]
    correct = 1. / np.sqrt(2)**3 * np.array([1] * 2 + [0] * 6)
    assert np.allclose(mps.wavefunction(), correct)

    # Apply |0><0| to the third qubit
    mps.apply_one_qudit_gate(
        pi0,
        2,
        ortho_after_non_unitary=False,
        renormalize_after_non_unitary=False
    )  # State: 1 / sqrt(2)**3 * |000>
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / 2. / np.sqrt(2))
    assert mps.bond_dimensions() == [1, 1]
    correct = 1. / np.sqrt(2) ** 3 * np.array([1] * 1 + [0] * 7)
    assert np.allclose(mps.wavefunction(), correct)


def test_apply_povm_bell_state_right_ortho_reduces_bond_dimension():
    """Tests applying a POVM + orthonormalizing the index to a bell state."""
    # Get the projector
    pi0 = computational_basis_projector(state=0)

    # Create an MPS in the Bell state
    n = 2
    mps = MPS(nqudits=n)  # State: |00>
    mps_operations = [
        MPSOperation(hgate(), (0,)),
        MPSOperation(cnot(), (0, 1))
    ]
    mps.apply(mps_operations)  # State: 1 / sqrt(2) |00> + |11>
    assert np.isclose(mps.norm(), 1.0)
    assert mps.bond_dimensions() == [2]
    wavefunction_before = mps.wavefunction()

    # Check that orthonormalization does nothing to the Bell state
    mps.orthonormalize_right_edge_of(node_index=0)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1.0)
    assert np.allclose(mps.wavefunction(), wavefunction_before)
    assert mps.bond_dimensions() == [2]

    # Apply |0><0| to the first qubit
    mps.apply_one_qudit_gate(
        pi0,
        0,
        ortho_after_non_unitary=False,
        renormalize_after_non_unitary=False
    )
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))
    correct = 1. / np.sqrt(2) * np.array([1., 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)
    assert mps.bond_dimensions() == [2]

    # Now do the orthonormalization to reduce the bond dimension
    mps.orthonormalize_right_edge_of(node_index=0)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))
    assert np.allclose(mps.wavefunction(), correct)
    assert mps.bond_dimensions() == [1]


def test_apply_povm_bell_state_left_ortho_reduces_bond_dimension():
    """Tests applying a POVM + orthonormalizing the index to a bell state."""
    # Get the projector
    pi0 = computational_basis_projector(state=0)

    # Create an MPS in the Bell state
    n = 2
    mps = MPS(nqudits=n)  # State: |00>
    mps_operations = [
        MPSOperation(hgate(), (0,)),
        MPSOperation(cnot(), (0, 1))
    ]
    mps.apply(mps_operations)  # State: 1 / sqrt(2) |00> + |11>
    assert np.isclose(mps.norm(), 1.0)
    assert mps.bond_dimensions() == [2]
    wavefunction_before = mps.wavefunction()

    # Check that orthonormalization does nothing to the Bell state
    mps.orthonormalize_left_edge_of(node_index=1)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1.0)
    assert np.allclose(mps.wavefunction(), wavefunction_before)
    assert mps.bond_dimensions() == [2]

    # Apply |0><0| to the second qubit
    mps.apply_one_qudit_gate(
        pi0,
        1,
        ortho_after_non_unitary=False,
        renormalize_after_non_unitary=False
    )
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))
    correct = 1. / np.sqrt(2) * np.array([1., 0., 0., 0.])
    assert np.allclose(mps.wavefunction(), correct)
    assert mps.bond_dimensions() == [2]

    # Now do the orthonormalization to reduce the bond dimension
    mps.orthonormalize_left_edge_of(node_index=1)
    assert mps.is_valid()
    assert np.isclose(mps.norm(), 1. / np.sqrt(2))
    assert np.allclose(mps.wavefunction(), correct)
    assert mps.bond_dimensions() == [1]


def test_orthonormalize_all_tensors_edge_cases():
    """Tests orthonormalizing all tensors in an MPS and ensures the MPS remains
    valid after.
    """
    for n in range(2, 8):
        for d in (2, 3, 4):
            mps = MPS(nqudits=n, qudit_dimension=d)
            correct = mps.wavefunction()
            for node_index in range(n - 1):
                mps.orthonormalize_right_edge_of(node_index)
                assert mps.is_valid()
                assert np.allclose(mps.wavefunction(), correct)
                assert np.isclose(mps.norm(), 1.)
            for node_index in range(1, n):
                mps.orthonormalize_left_edge_of(node_index)
                assert mps.is_valid()
                assert np.allclose(mps.wavefunction(), correct)
                assert np.isclose(mps.norm(), 1.)


def test_renormalize_after_non_unitary():
    """Applies non-unitary POVMs and tests that norm remains the same."""
    nqubits = 6
    depth = 10
    mps = MPS(nqudits=nqubits)
    pi0 = computational_basis_projector(state=0, dim=2)
    for _ in range(depth):
        mps.r(-1)
        mps.sweep_cnots_left_to_right()
        assert np.isclose(mps.norm(), 1.)
        for i in range(nqubits):
            mps.apply_one_qudit_gate(
                gate=pi0,
                node_index=i,
                ortho_after_non_unitary=True,
                renormalize_after_non_unitary=True
            )
            assert np.isclose(mps.norm(), 1.)


@pytest.mark.parametrize("chi", [16, 32, 64, 128])
def test_max_bond_dimension_not_surpassed(chi: int):
    """Applies operations with a max chi value and ensures the bond dimensions
    of the tensors never gets bigger than chi.
    """
    nqubits = 10
    depth = 10
    mps = MPS(nqudits=nqubits, qudit_dimension=2)

    singles = (hgate(), xgate(), zgate())
    czgate = cphase(exp=0.5)

    # Apply operations
    for _ in range(depth):
        for i in range(nqubits):
            gate = np.random.choice(singles)
            op = MPSOperation(gate, (i,))
            mps.apply(op)

        for i in range(nqubits):
            other_qubits = list(set(range(nqubits)) - {i})
            j = np.random.choice(other_qubits)
            op = MPSOperation(czgate, (i, j))
            mps.apply(op, maxsvals=chi)

        assert all(bond_dimension <= chi
                   for bond_dimension in mps.bond_dimensions())


def test_equal():
    """Tests checking equality of MPS."""
    for n in (2, 3, 5, 10):
        for d in (2, 3, 5, 10):
            mps1 = MPS(nqudits=n, qudit_dimension=d)
            mps2 = MPS(nqudits=n, qudit_dimension=d)
            assert mps1 == mps1
            assert mps2 == mps2
            assert mps1 == mps2

            if d == 2:
                mps1.apply(MPSOperation(xgate(), 0))
                assert mps1 != mps2

                mps2.apply(MPSOperation(xgate(), 0))
                assert mps1 == mps2


def test_equal_different_prefixes():
    """Tests identical MPS with different tensor names are still equal."""
    mps1 = MPS(nqudits=10, qudit_dimension=2, tensor_prefix="mps1_")
    mps2 = MPS(nqudits=10, qudit_dimension=2, tensor_prefix="mps2_")
    assert mps1 == mps2


def test_copy():
    """Tests copying an MPS by calling copy(MPS)."""
    for n in (2, 3, 5, 10):
        for d in (2, 3, 5, 10):
            mps = MPS(nqudits=n, qudit_dimension=d)
            mps_copy = copy(mps)
            assert mps_copy is not mps
            assert mps_copy == mps


def test_copy_method():
    """Tests copying an MPS by calling MPS.copy()."""
    for n in (2, 3, 5, 10):
        for d in (2, 3, 5, 10):
            mps = MPS(nqudits=n, qudit_dimension=d)
            mps_copy = mps.copy()
            assert mps_copy is not mps
            assert mps_copy == mps


def test_expectation_two_qubit_mps():
    """Tests some expectation values for a two-qubit MPS."""
    # |00>
    mps = MPS(nqudits=2)
    mps_copy = mps.copy()

    # <00|HI|00> = 1 / sqrt(2)
    h0 = MPSOperation(hgate(), 0)
    assert np.isclose(mps.expectation(h0), 1. / np.sqrt(2))
    assert mps == mps_copy

    # <00|XI|00> = 0
    x0 = MPSOperation(xgate(), 0)
    assert np.isclose(mps.expectation(x0), 0.)
    assert mps == mps_copy

    # <10|HI|10> = - 1 / sqrt(2)
    mps.apply(MPSOperation(xgate(), 0))
    assert np.isclose(mps.expectation(h0), -1. / np.sqrt(2))


def test_dagger_simple():
    """Tests taking the dagger of an MPS."""
    wavefunction = np.array([1j, 0., 0., 0.])
    mps = MPS.from_wavefunction(wavefunction, nqudits=2, qudit_dimension=2)
    assert np.allclose(mps.wavefunction(), wavefunction)
    mps.dagger()
    assert np.allclose(mps.wavefunction(), wavefunction.conj().T)


def test_dagger_random_qubit_wavefunctions():
    """Tests taking the dagger of an MPS created from random wavefunctions."""
    np.random.seed(10)
    for n in (2, 3, 5, 10):
        for _ in range(20):
            wavefunction = np.random.randn(2**n) + np.random.randn(2**n) * 1j
            wavefunction /= np.linalg.norm(wavefunction, ord=2)
            mps = MPS.from_wavefunction(wavefunction, nqudits=n)
            assert np.allclose(mps.wavefunction(), wavefunction)
            mps.dagger()
            assert np.allclose(mps.wavefunction(), wavefunction.conj().T)


def test_reduced_density_matrix_simple():
    """Tests computing the reduced density matrix of both sites of a two-qubit
    MPS product states.
    """
    # State: |00>
    mps = MPS(nqudits=2, qudit_dimension=2)
    for i in (0, 1):
        rdm = mps.reduced_density_matrix(node_indices=i)
        correct = np.array([[1., 0.], [0., 0.]])
        assert np.allclose(rdm, correct)
        assert mps == MPS(nqudits=2, qudit_dimension=2)

    # State: |10>
    mps.apply(MPSOperation(xgate(), 0))
    rdm = mps.reduced_density_matrix(node_indices=0)
    correct = np.array([[0., 0.], [0., 1.]])
    assert np.allclose(rdm, correct)

    rdm = mps.reduced_density_matrix(node_indices=1)
    correct = np.array([[1., 0.], [0., 0.]])
    assert np.allclose(rdm, correct)

    # State |11>
    mps.apply(MPSOperation(xgate(), 1))
    rdm = mps.reduced_density_matrix(node_indices=0)
    correct = np.array([[0., 0.], [0., 1.]])
    assert np.allclose(rdm, correct)

    rdm = mps.reduced_density_matrix(node_indices=1)
    correct = np.array([[0., 0.], [0., 1.]])
    assert np.allclose(rdm, correct)


def test_reduced_density_matrix_invalid_indices():
    """Tests the correct errors are raised for invalid indices."""
    mps = MPS(nqudits=2)

    with pytest.raises(IndexError):
        mps.reduced_density_matrix(node_indices=-1)

    with pytest.raises(IndexError):
        mps.reduced_density_matrix(node_indices=22)

    with pytest.raises(ValueError):
        mps.reduced_density_matrix(node_indices=[0, 0])


def test_reduced_density_matrix_two_qubits():
    """Tests computing the reduced density matrix for a two-qubit MPS with
    random wavefunctions.
    """
    np.random.seed(3)
    for _ in range(50):
        wavefunction = np.random.randn(4) + np.random.randn(4) * 1j
        wavefunction /= np.linalg.norm(wavefunction)
        mps = MPS.from_wavefunction(wavefunction, nqudits=2)

        for i in (0, 1):
            correct = density_matrix_from_state_vector(
                state=wavefunction, indices=[i]
            )
            rdm = mps.reduced_density_matrix(node_indices=i)
            assert np.allclose(rdm, correct)
            assert np.allclose(mps.wavefunction(), wavefunction)


def test_density_matrix_three_qubits():
    """Tests computing the full density matrix on a three-qubit MPS."""
    np.random.seed(5)
    for _ in range(50):
        wavefunction = np.random.randn(8) + np.random.randn(8) * 1j
        wavefunction /= np.linalg.norm(wavefunction)
        mps = MPS.from_wavefunction(wavefunction, nqudits=3)

        for i in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            correct = density_matrix_from_state_vector(
                state=wavefunction, indices=i
            )
            rdm = mps.reduced_density_matrix(node_indices=i)
            assert np.allclose(rdm, correct)

            correct = density_matrix_from_state_vector(
                state=wavefunction, indices=tuple(reversed(i))
            )
            rdm = mps.reduced_density_matrix(node_indices=tuple(reversed(i)))
            assert np.allclose(rdm, correct)
            assert np.allclose(mps.wavefunction(), wavefunction)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_qubit_mps_single_site_density_matrices(n: int):
    """Tests computing single-site partial density matrices on qubit MPS."""
    np.random.seed(1)
    wavefunction = np.random.randn(2**n) + np.random.randn(2**n) * 1j
    wavefunction /= np.linalg.norm(wavefunction)
    mps = MPS.from_wavefunction(wavefunction, nqudits=n)

    site = [int(np.random.choice(range(n)))]
    rdm = mps.reduced_density_matrix(node_indices=site)
    correct = density_matrix_from_state_vector(
        state=wavefunction, indices=site
    )
    assert np.allclose(rdm, correct)
    assert np.allclose(mps.wavefunction(), wavefunction)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_qubit_mps_multi_site_density_matrices(n: int):
    """Tests computing partial density matrices."""
    np.random.seed(1)
    for _ in range(50):
        wavefunction = np.random.randn(2**n) + np.random.randn(2**n) * 1j
        wavefunction /= np.linalg.norm(wavefunction)
        mps = MPS.from_wavefunction(wavefunction, nqudits=n)

        size = np.random.randint(low=1, high=n)
        qubits = list(np.random.choice(range(n), size=size, replace=False))
        sites = [int(q) for q in qubits]
        rdm = mps.reduced_density_matrix(node_indices=sites)
        correct = density_matrix_from_state_vector(
            state=wavefunction, indices=sites
        )
        assert np.allclose(rdm, correct)
        assert np.allclose(mps.wavefunction(), wavefunction)


def test_sample_simple():
    """Sampling from the |00> MPS."""
    mps = MPS(nqudits=2)
    string = mps.sample(nsamples=10)
    print(string)
    assert False


def test_sample_uniform():
    """Sampling from the |++...+> MPS."""
    n = 3
    mps = MPS(nqudits=n)
    mps.apply(
        [MPSOperation(hgate(), i) for i in range(n)]
    )
    samples = mps.sample(nsamples=10)
    print(samples)
    assert False


def test_sample_from_wavefunction():
    """Tests sampling from an MPS with a known wavefunction."""
    np.random.seed(1)
    n = 3
    wavefunction = np.random.randn(2**n) + np.random.randn(2**n) * 1j
    wavefunction /= np.linalg.norm(wavefunction)
    mps = MPS.from_wavefunction(wavefunction, nqudits=n)
    print("Distribution")
    print(np.round([abs(alpha)**2 for alpha in wavefunction], 2))

    samples = mps.sample(nsamples=10)
    print(samples)
    assert False
