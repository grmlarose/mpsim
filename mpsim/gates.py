"""Declarations of single qubit and two-qubit gates."""

from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

import tensornetwork as tn

# Common single qubit states as np.ndarray objects
zero_state = np.array([1.0, 0.0], dtype=np.complex64)
one_state = np.array([0.0, 1.0], dtype=np.complex64)
plus_state = 1.0 / np.sqrt(2) * (zero_state + one_state)


# Common single qubit gates as np.ndarray objects
_hmatrix = (
    1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64)
)
_imatrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex64)
_xmatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
_ymatrix = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex64)
_zmatrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)


# Common single qubit gates as tn.Node objects
# Note that functions are used because TensorNetwork connect/contract
# functions modify Node objects
def igate() -> tn.Node:
    """Returns a single qubit identity gate."""
    return tn.Node(deepcopy(_imatrix), name="igate")


def xgate() -> tn.Node:
    """Returns a Pauli X (NOT) gate."""
    return tn.Node(deepcopy(_xmatrix), name="xgate")


def ygate() -> tn.Node:
    """Returns a Pauli Y gate."""
    return tn.Node(deepcopy(_ymatrix), name="ygate")


def zgate() -> tn.Node:
    """Returns a Pauli Z gate."""
    return tn.Node(deepcopy(_zmatrix), name="zmat")


def hgate() -> tn.Node:
    """Returns a Hadamard gate."""
    return tn.Node(deepcopy(_hmatrix), name="hgate")


def rgate(seed: Optional[int] = None, angle_scale: float = 1.0):
    """Returns the random single qubit gate described in
    https://arxiv.org/abs/2002.07730.
    
    Args:
        seed: Seed for random number generator.
        angle_scale: Floating point value to scale angles by. Default 1.
    
    """
    if seed:
        np.random.seed(seed)

    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi
    mx = np.sin(alpha) * np.cos(phi)
    my = np.sin(alpha) * np.sin(phi)
    mz = np.cos(alpha)
    
    theta *= angle_scale

    # Get the unitary
    unitary = expm(
        -1j * theta * (mx * _xmatrix + my * _ymatrix * mz * _zmatrix)
    )

    # TODO: Note to Guifre diagonal elements of this unitary are always real,
    #  and off-diagonal elements are imaginary
    return tn.Node(unitary)


def computational_basis_projector(state: int, dim: int = 2) -> tn.Node:
    """Returns a projector onto a computational basis state which acts on a
    single qudit of dimension dim.

    Args:
        state: Basis state to project onto.
        dim: Dimension of the qudit. Default is two for qubits.

    Raises:
        ValueError: If state < 0, dim < 0, or state >= dim.
    """
    if state < 0:
        raise ValueError(f"Argument state should be positive but is {state}.")

    if dim < 0:
        raise ValueError(f"Argument dim should be positive but is {dim}.")

    if state >= dim:
        raise ValueError(
            f"Requires state < dim but state = {state} and dim = {dim}"
        )
    projector = np.zeros((dim, dim))
    projector[state, state] = 1.
    return tn.Node(projector, name=f"|{state}><{state}|")


# Common two qubit gates as np.ndarray objects
_cnot_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
_cnot_matrix = np.reshape(_cnot_matrix, newshape=(2, 2, 2, 2))
_swap_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_swap_matrix = np.reshape(_swap_matrix, newshape=(2, 2, 2, 2))


# Common two qubit gates as tn.Node objects
def cnot() -> tn.Node:
    return tn.Node(deepcopy(_cnot_matrix), name="cnot")


def swap() -> tn.Node:
    return tn.Node(deepcopy(_swap_matrix), name="swap")


def cphase(exp: float) -> tn.Node:
    matrix = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., np.exp(1j * 2 * np.pi * exp)]
    ], dtype=np.complex64)
    matrix = np.reshape(matrix, newshape=(2, 2, 2, 2))
    return tn.Node(matrix, name="cphase")


def random_two_qubit_gate(seed: Optional[int] = None) -> tn.Node:
    """Returns a random two-qubit gate.

    Args:
        seed: Seed for random number generator.
    """
    if seed:
        np.random.seed(seed)
    unitary = unitary_group.rvs(dim=4)
    unitary = np.reshape(unitary, newshape=(2, 2, 2, 2))
    return tn.Node(deepcopy(unitary), name="R2Q")
