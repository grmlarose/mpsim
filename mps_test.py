"""Unit tests for inital MPS states."""

import pytest

import numpy as np
import tensornetwork as tn

import mps.mps as mps


def test_mps_one_qubit():
    """Ensures an error is raised if the number of qubits is less than two."""
    with pytest.raises(ValueError):
        mps.get_zero_state_mps(1)


def test_get_wavefunction_simple():
    """Tests getting the wavefunction of a simple MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=3)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    assert isinstance(wavefunction, np.ndarray)
    assert wavefunction.shape == (8,)
    correct = np.array([1.] + [0.] * 7, dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_get_wavefunction_deosnt_modify_mps():
    """Tests that getting the wavefunction of an MPS doesn't affect the actual MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=2)
    nodes_list, _ = tn.copy(mpslist)
    _ = mps.get_wavefunction_of_mps(mpslist)
    a, b = mpslist
    assert len(a.edges) == 2
    assert len(a.get_all_nondangling()) == 1
    assert len(a.get_all_dangling()) == 1
    assert len(b.edges) == 2
    assert len(b.get_all_nondangling()) == 1
    assert len(b.get_all_dangling()) == 1


def test_correctness_of_initial_product_state_two_qubits():
    """Tests that the contracted MPS is indeed the all zero state for two qubits."""
    lq, _ = mps.get_zero_state_mps(nqubits=2)
    wavefunction_node = tn.contract(lq[1])
    wavefunction = np.reshape(wavefunction_node.tensor, newshape=(4,))
    correct = np.array([1, 0, 0, 0], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_correctness_of_initial_product_state():
    """Tests that the contracted MPS is indeed the all zero state for multiple qubits."""
    for n in range(3, 10):
        mpslist = mps.get_zero_state_mps(n)
        wavefunction = mps.get_wavefunction_of_mps(mpslist)
        correct = np.array([1] + [0] * (2**n - 1), dtype=np.complex64)
        assert np.array_equal(wavefunction, correct)


@pytest.mark.parametrize(["gate", "expected"],
                         [(mps.xgate(), mps.one_state),
                          (mps.hgate(), mps.plus_state),
                          (mps.zgate(), mps.zero_state),
                          (mps.igate(), mps.zero_state)])
def test_apply_oneq_gate_xgate(gate, expected):
    """Tests application of a single qubit gate to several MPS."""
    for n in range(2, 8):
        for j in range(n):
            mpslist = mps.get_zero_state_mps(n)
            mps.apply_one_qubit_gate(gate, j, mpslist)
            final_state = np.reshape(mpslist[j].tensor, newshape=(2,))
            assert np.array_equal(final_state, expected)


def test_apply_oneq_gate_to_all():
    """Tests correctness for final wavefunction after applying a NOT gate to all qubits in a two-qubit MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate_to_all(mps.xgate(), mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_apply_oneq_gate_to_all_hadamard():
    """Tests correctness for final wavefunction after applying a Hadamard gate to all qubits in a five-qubit MPS."""
    n = 5
    mpslist = mps.get_zero_state_mps(nqubits=5)
    mps.apply_one_qubit_gate_to_all(mps.hgate(), mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = 1 / 2**(n / 2) * np.ones(2**n)
    assert np.allclose(wavefunction, correct)


def test_apply_twoq_cnot_two_qubits():
    """Tests for correctness of final wavefunction after applying a CNOT to a two-qubit MPS."""
    # In the following tests, the first qubit is always the control qubit.
    # Check that CNOT|10> = |11>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|00> = |00>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([1., 0., 0., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|01> = |01>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate(mps.xgate(), 1, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 1., 0., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|11> = |10>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate_to_all(mps.xgate(), mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 1., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_apply_twoq_cnot_two_qubits_flipped_control_and_target():
    """Tests for correctness of final wavefunction after applying a CNOT to a two-qubit MPS."""
    # In the following tests, the first qubit is always the target qubit.
    # Check that CNOT|10> = |10>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 1, 0, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 1., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|00> = |00>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_two_qubit_gate(mps.cnot(), 1, 0, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([1., 0., 0., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|01> = |11>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate(mps.xgate(), 1, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 1, 0, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)

    # Check that CNOT|11> = |01>
    mpslist = mps.get_zero_state_mps(nqubits=2)
    mps.apply_one_qubit_gate_to_all(mps.xgate(), mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 1, 0, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 1., 0., 0.], dtype=np.complex64)
    assert np.array_equal(wavefunction, correct)


def test_apply_twoq_identical_indices_raises_error():
    """Tests that a two-qubit gate application with identical indices raises an error."""
    mps2q = mps.get_zero_state_mps(nqubits=2)
    mps3q = mps.get_zero_state_mps(nqubits=3)
    mps9q = mps.get_zero_state_mps(nqubits=9)
    with pytest.raises(ValueError):
        for mpslist in (mps2q, mps3q, mps9q):
            mps.apply_two_qubit_gate(mps.cnot(), 0, 0, mpslist)


def test_apply_twoq_non_adjacent_indices_raises_error():
    """Tests that a two-qubit gate application with non-adjacent indices raises an error."""
    n = 10
    mpslist = mps.get_zero_state_mps(nqubits=n)
    for (a, b) in [(0, 2), (0, 9), (4, 6), (2, 7)]:
        with pytest.raises(ValueError):
            mps.apply_two_qubit_gate(mps.cnot(), a, b, mpslist)


def test_apply_twoq_cnot_four_qubits_interior_qubits():
    """Tests with a CNOT on four qubits acting on "interior" qubits."""
    mpslist = mps.get_zero_state_mps(nqubits=4)             # State: |0000>
    mps.apply_one_qubit_gate(mps.xgate(), 1, mpslist)       # State: |0100>
    mps.apply_two_qubit_gate(mps.cnot(), 1, 2, mpslist)     # State: Should be |0110>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.array_equal(wavefunction, correct)

    mpslist = mps.get_zero_state_mps(nqubits=4)             # State: |0000>
    mps.apply_two_qubit_gate(mps.cnot(), 1, 2, mpslist)     # State: Should be |0000>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.array_equal(wavefunction, correct)


def test_apply_twoq_cnot_four_qubits_edge_qubits():
    """Tests with a CNOT on four qubits acting on "edge" qubits."""
    mpslist = mps.get_zero_state_mps(nqubits=4)             # State: |0000>
    mps.apply_one_qubit_gate(mps.xgate(), 2, mpslist)       # State: |0010>
    mps.apply_two_qubit_gate(mps.cnot(), 2, 3, mpslist)     # State: Should be |0011>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.array_equal(wavefunction, correct)

    mpslist = mps.get_zero_state_mps(nqubits=4)             # State: |0000>
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)       # State: |1000>
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)     # State: Should be |1100>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    print(wavefunction)
    correct = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    assert np.array_equal(wavefunction, correct)


def test_apply_twoq_cnot_five_qubits_all_combinations():
    """Tests applying a CNOT to a five qubit MPS with all combinations of indices.

    That is, CNOT_01, CNOT_02, CNOT_03, ..., CNOT_24, CNOT_34 where the first number is
    the control qubit and the second is the target qubit.
    """
    n = 5
    indices = [(a, a + 1) for a in range(n - 1)]
    print(indices)

    for (a, b) in indices:
        # Apply the gates
        mpslist = mps.get_zero_state_mps(nqubits=n)
        mps.apply_one_qubit_gate(mps.xgate(), a, mpslist)
        mps.apply_two_qubit_gate(mps.cnot(), a, b, mpslist)
        wavefunction = mps.get_wavefunction_of_mps(mpslist)
        # Get the correct wavefunction
        correct = np.zeros((2**n,))
        bits = ["0"] * n
        bits[a] = "1"
        bits[b] = "1"
        correct[int("".join(bits), 2)] = 1.
        assert np.array_equal(wavefunction, correct)


@pytest.mark.parametrize(["indices"],
                         [[(0, 1)], [(1, 0)]])
def test_apply_twoq_swap_two_qubits(indices):
    """Tests swapping two qubits in a two-qubit MPS."""
    mpslist = mps.get_zero_state_mps(nqubits=2)                 # State: |00>
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)           # State: |10>
    mps.apply_two_qubit_gate(mps.swap(), *indices, mpslist)     # State: |01>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 1., 0., 0.])
    assert np.array_equal(wavefunction, correct)

    mpslist = mps.get_zero_state_mps(nqubits=2)                 # State: |00>
    mps.apply_two_qubit_gate(mps.swap(), *indices, mpslist)     # State: |00>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([1., 0., 0., 0.])
    assert np.array_equal(wavefunction, correct)

    mpslist = mps.get_zero_state_mps(nqubits=2)                 # State: |00>
    mps.apply_one_qubit_gate(mps.xgate(), 1, mpslist)           # State: |01>
    mps.apply_two_qubit_gate(mps.swap(), *indices, mpslist)     # State: |10>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 1., 0.])
    assert np.array_equal(wavefunction, correct)

    mpslist = mps.get_zero_state_mps(nqubits=2)                 # State: |00>
    mps.apply_one_qubit_gate_to_all(mps.xgate(), mpslist)       # State: |11>
    mps.apply_two_qubit_gate(mps.swap(), *indices, mpslist)     # State: |11>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 0., 0., 1.])
    assert np.array_equal(wavefunction, correct)


def test_apply_swap_five_qubits():
    """Tests applying a swap gate to an MPS with five qubits."""
    n = 5
    for i in range(n - 1):
        mpslist = mps.get_zero_state_mps(nqubits=n)
        mps.apply_one_qubit_gate(mps.xgate(), i, mpslist)
        mps.apply_two_qubit_gate(mps.swap(), i, i + 1, mpslist)
        wavefunction = mps.get_wavefunction_of_mps(mpslist)
        # Get the correct wavefunction
        correct = np.zeros((2 ** n,))
        bits = ["0"] * n
        bits[i + 1] = "1"
        correct[int("".join(bits), 2)] = 1.
        assert np.array_equal(wavefunction, correct)


def test_qubit_hopping():
    """Tests "hopping" a qubit with a sequence of swap gates."""
    n = 8
    mpslist = mps.get_zero_state_mps(nqubits=n)
    mps.apply_one_qubit_gate(mps.hgate(), 0, mpslist)
    for i in range(1, n - 1):
        mps.apply_two_qubit_gate(mps.swap(), i, i + 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.zeros(2**n)
    correct[0] = correct[2**(n - 1)] = 1. / np.sqrt(2)
    assert np.allclose(wavefunction, correct)


def test_bell_state():
    """Tests for wavefunction correctness after preparing a Bell state."""
    n = 2
    mpslist = mps.get_zero_state_mps(nqubits=n)
    mps.apply_one_qubit_gate(mps.hgate(), 0, mpslist)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = 1. / np.sqrt(2) * np.array([1., 0., 0., 1.])
    assert np.allclose(wavefunction, correct)


def test_twoq_gates_in_succession():
    """Tests for wavefunction correctness after applying a series of two-qubit gates."""
    n = 2
    mpslist = mps.get_zero_state_mps(nqubits=n)
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)       # State: |10>
    mps.apply_two_qubit_gate(mps.cnot(), 1, 0, mpslist)     # State: |10>
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, mpslist)     # State: |11>
    mps.apply_one_qubit_gate(mps.xgate(), 0, mpslist)       # State: |01>
    wavefunction = mps.get_wavefunction_of_mps(mpslist)
    correct = np.array([0., 1., 0., 0.])
    assert np.array_equal(wavefunction, correct)


def test_left_vs_right_canonical_two_qubit_one_gate():
    """Performs a two-qubit gate keeping left-canonical and right-canonical,
    checks for equality in final wavefunction.
    """
    n = 2
    lmps = mps.get_zero_state_mps(nqubits=n)
    rmps = mps.get_zero_state_mps(nqubits=n)
    mps.apply_one_qubit_gate(mps.xgate(), 0, lmps)
    mps.apply_one_qubit_gate(mps.xgate(), 0, rmps)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, lmps, keep_left_canonical=True)
    mps.apply_two_qubit_gate(mps.cnot(), 0, 1, rmps, keep_left_canonical=False)
    lwavefunction = mps.get_wavefunction_of_mps(lmps)
    rwavefunction = mps.get_wavefunction_of_mps(rmps)
    cwavefunction = np.array([0., 0., 0., 1.])
    assert np.array_equal(lwavefunction, cwavefunction)
    assert np.array_equal(rwavefunction, cwavefunction)


def test_prepare_ghz_state():
    """Tests preparation of the GHZ state on four qubits."""
    pass
