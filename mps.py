"""Code for Matrix Product State initialization, manipulation, and operations."""

from typing import List

import numpy as np
import tensornetwork as tn


# Common single qubit states as np.ndarray objects
zero_state = np.array([1., 0.], dtype=np.complex64)
one_state = np.array([0., 1.], dtype=np.complex64)
plus_state = 1. / np.sqrt(2) * (zero_state + one_state)


# Common single qubit gates as tn.Node objects
_hmatrix = 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]], dtype=np.complex64)
hgate = tn.Node(_hmatrix, name="hgate")

_imatrix = np.array([[1., 0.], [0., 1.]], dtype=np.complex64)
igate = tn.Node(_imatrix, name="igate")

_xmatrix = np.array([[0., 1.], [1., 0.]], dtype=np.complex64)
xgate = tn.Node(_xmatrix, name="xgate")

_ymatrix = np.array([[0., -1j], [1j, 0.]], dtype=np.complex64)
ygate = tn.Node(_ymatrix, name="ygate")

_zmatrix = np.array([[1., 0.], [0., -1.]], dtype=np.complex64)
zgate = tn.Node(_zmatrix, name="zgate")


# Common two qubit gates as tn.Node objects
_cnot_matrix = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.],
                         [0., 0., 1., 0.]])
_cnot_matrix = np.reshape(_cnot_matrix, newshape=(2, 2, 2, 2))
cnot = tn.Node(_cnot_matrix, name="cnot")


def get_zero_state_mps(nqubits: int, tensor_prefix: str = "q") -> List[tn.Node]:
    """Returns a list of tensors in an MPS which define the all zero state on n qubits.

    The MPS has the following structure (shown for six qubits):

        @ ---- @ ---- @ ---- @ ---- @ ---- @
        |      |      |      |      |      |

    Virtual indices have bond dimension one and physical indices have bond dimension 2.

    Args:
        nqubits: Number of qubits in the all zero state.
        tensor_prefix: Prefix for tensors. The full name is prefix + numerical index,
                       numbered from left to right starting with zero.
    """
    if nqubits < 2:
        raise ValueError(f"Number of qubits must be greater than 2 but is {nqubits}.")

    # Get nodes on the interior
    nodes = [tn.Node(
        np.array([[[1.]], [[0, ]]], dtype=np.complex64), name=tensor_prefix + str(x + 1)
    ) for x in range(nqubits - 2)]

    # Get nodes on the end
    nodes.insert(0, tn.Node(np.array([[1.], [0, ]], dtype=np.complex64), name=tensor_prefix + str(0)))
    nodes.append(tn.Node(np.array([[1.], [0, ]], dtype=np.complex64), name=tensor_prefix + str(nqubits - 1)))

    # Connect edges between middle nodes
    for i in range(1, nqubits - 2):
        tn.connect(nodes[i].get_edge(2), nodes[i + 1].get_edge(1))

    # Connect end nodes to the adjacent middle nodes
    if nqubits < 3:
        tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
    else:
        tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
        tn.connect(nodes[-1].get_edge(1), nodes[-2].get_edge(2))

    return nodes


def get_wavefunction_of_mps(mps: List[tn.Node]) -> np.array:
    """Returns the wavefunction of a valid MPS as a (potentially giant) vector by contracting all virtual indices.

    NOTE: Calling this function "destroys" the MPS because apparently it's not possible to copy a list of
    tensors in TensorNetwork...

    Args:
        mps: List of tn.Node objects defining a valid MPS.
    """
    n = len(mps)
    fin = mps.pop(0)
    for node in mps:
        fin = tn.contract_between(fin, node)

    # Make sure all edges are free
    if set(fin.get_all_dangling()) != set(fin.get_all_edges()):
        raise ValueError("Invalid MPS.")

    return np.reshape(fin.tensor, newshape=(2 ** n))


def apply_one_qubit_gate(gate: tn.Node, index: int, mpslist: List[tn.Node]) -> None:
    """Modifies the input mpslist in place by applying a single qubit gate to a specified node.

    Args:
        gate: Single qubit gate to apply. A tensor with two free indices.
        index: Index of tensor (qubit) in the mpslist to apply the single qubit gate to.
        mpslist: List of tn.Node objects representing a valid MPS.
    """
    # TODO: Check that mpslist defines a valid mps.

    if index not in range(len(mpslist)):
        raise ValueError(f"Input tensor index={index} is out of bounds for the input mpslist.")

    if len(gate.get_all_dangling()) != 2 or len(gate.get_all_nondangling()) != 0:
        raise ValueError("Single qubit gate must have two free edges and zero connected edges.")

    # Connect the MPS and gate edges
    mps_edge = list(mpslist[index].get_all_dangling())[0]
    gate_edge = gate[0]
    connected = tn.connect(mps_edge, gate_edge)

    # Contract the edge to get the new tensor
    new = tn.contract(connected)
    mpslist[index] = new


def apply_one_qubit_gate_to_all(gate: tn.Node, mpslist: List[tn.Node]):
    """Modifies the input mpslist in place by applying a single qubit gate to all tensors in the MPS.

    Args:
        gate: Single qubit gate to apply. A tensor with two free indices.
        mpslist: List of tn.Node objects representing a valid MPS.
    """
    for i in range(len(mpslist)):
        apply_one_qubit_gate(gate, i, mpslist)


def apply_two_qubit_gate(gate: tn.Node, indexA: int, indexB: int, mpslist: List[tn.Node]) -> None:
    """Modifies the input mpslist in place by applying a single qubit gate to a specified node.

    Args:
        gate: Single qubit gate to apply.
        indexA: Index of first tensor (qubit) in the mpslist to apply the single qubit gate to.
        indexB: Index of second tensor (qubit) in the mpslist to apply the single qubit gate to.
        mpslist: List of tn.Node's representing a valid MPS.
    """
    # TODO: Check that mpslist defines a valid mps.

    if indexA not in range(len(mpslist)) or indexB not in range(len(mpslist)):
        raise ValueError(f"Input tensor indices={(indexA, indexB)} are out of bounds for the input mpslist.")

    if len(gate.get_all_dangling()) != 4 or len(gate.get_all_nondangling()) != 0:
        raise ValueError("Two qubit gate must have four free edges and zero connected edges.")

    # Connect the MPS and gate edges
    mpsA_edge = list(mpslist[indexA].get_all_dangling())[0]
    mpsB_edge = list(mpslist[indexB].get_all_dangling())[0]
    gateA_edge = gate[0]
    gateB_edge = gate[1]
    connectedA = tn.connect(mpsA_edge, gateA_edge)
    connectedB = tn.connect(mpsB_edge, gateB_edge)

    # Contract the edges to get the new tensors
    newA = tn.contract(connectedA)
    newB = tn.contract(connectedB)

    mpslist[indexA] = newA
    mpslist[indexB] = newB
