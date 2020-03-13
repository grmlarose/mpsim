"""Code for Matrix Product State initialization, manipulation, and operations."""

from copy import deepcopy
from typing import List

import numpy as np
import tensornetwork as tn


# Common single qubit states as np.ndarray objects
zero_state = np.array([1., 0.], dtype=np.complex64)
one_state = np.array([0., 1.], dtype=np.complex64)
plus_state = 1. / np.sqrt(2) * (zero_state + one_state)


# Common single qubit gates as np.ndarray objects
_hmatrix = 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]], dtype=np.complex64)
_imatrix = np.array([[1., 0.], [0., 1.]], dtype=np.complex64)
_xmatrix = np.array([[0., 1.], [1., 0.]], dtype=np.complex64)
_ymatrix = np.array([[0., -1j], [1j, 0.]], dtype=np.complex64)
_zmatrix = np.array([[1., 0.], [0., -1.]], dtype=np.complex64)


# Common single qubit gates as tn.Node objects
# Note that functions are used because TensorNetwork connect/contract functions modify Node objects
def igate():
    return tn.Node(deepcopy(_imatrix), name="igate")


def xgate():
    return tn.Node(deepcopy(_xmatrix), name="xgate")


def ygate():
    return tn.Node(deepcopy(_ymatrix), name="ygate")


def zgate():
    return tn.Node(deepcopy(_zmatrix), name="zmat")


def hgate():
    return tn.Node(deepcopy(_hmatrix), name="hgate")


# Common two qubit gates as np.ndarray objects
_cnot_matrix = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.],
                         [0., 0., 1., 0.]])
_cnot_matrix = np.reshape(_cnot_matrix, newshape=(2, 2, 2, 2))
_swap_matrix = np.array([[1., 0., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.]])
_swap_matrix = np.reshape(_swap_matrix, newshape=(2, 2, 2, 2))


# Common two qubit gates as tn.Node objects
def cnot():
    return tn.Node(deepcopy(_cnot_matrix), name="cnot")


def swap():
    return tn.Node(deepcopy(_swap_matrix), name="swap")


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


def get_wavefunction_of_mps(mpslist: List[tn.Node]) -> np.array:
    """Returns the wavefunction of a valid MPS as a (potentially giant) vector by contracting all virtual indices.

    NOTE: Calling this function "destroys" the MPS because apparently it's not possible to copy a list of
    tensors in TensorNetwork...

    Args:
        mpslist: List of tn.Node objects defining a valid MPS.
    """
    if not is_valid(mpslist):
        raise ValueError("Input mpslist does not define a valid MPS.")

    # Replicate the mps
    mpslist, _ = tn.copy(mpslist)
    mpslist = list(mpslist.values())
    n = len(mpslist)
    fin = mpslist.pop(0)
    for node in mpslist:
        fin = tn.contract_between(fin, node)

    # Make sure all edges are free
    if set(fin.get_all_dangling()) != set(fin.get_all_edges()):
        raise ValueError("Invalid MPS.")

    return np.reshape(fin.tensor, newshape=(2**n))


def is_valid(mpslist: List[tn.Node]) -> bool:
    """Returns true if the mpslist defines a valid MPS, else False.

    A valid MPS satisfies the following criteria:
        (1) At least two tensors.
        (2) Every tensor has exactly one free (dangling) edge.
        (3) Every tensor has connected edges to its nearest neighbor(s).
    """
    if len(mpslist) < 2:
        return False

    for (i, tensor) in enumerate(mpslist):
        # Exterior nodes
        if i == 0 or i == len(mpslist) - 1:
            if len(tensor.get_all_dangling()) != 1:
                return False
            if len(tensor.get_all_nondangling()) != 1:
                return False
        # Interior nodes
        else:
            if len(tensor.get_all_dangling()) != 1:
                return False
            if len(tensor.get_all_nondangling()) != 2:
                return False

        if i < len(mpslist) - 1:
            try:
                tn.check_connected((mpslist[i], mpslist[i + 1]))
            except ValueError:
                print(f"Nodes at index {i} and {i + 1} are not connected.")
                return False
    return True


def apply_one_qubit_gate(gate: tn.Node, index: int, mpslist: List[tn.Node]) -> None:
    """Modifies the input mpslist in place by applying a single qubit gate to a specified node.

    Args:
        gate: Single qubit gate to apply. A tensor with two free indices.
        index: Index of tensor (qubit) in the mpslist to apply the single qubit gate to.
        mpslist: List of tn.Node objects representing a valid MPS.
    """
    if not is_valid(mpslist):
        raise ValueError("Input mpslist does not define a valid MPS.")

    if index not in range(len(mpslist)):
        raise ValueError(f"Input tensor index={index} is out of bounds for the input mpslist.")

    if len(gate.get_all_dangling()) != 2 or len(gate.get_all_nondangling()) != 0:
        raise ValueError("Single qubit gate must have two free edges and zero connected edges.")

    # Connect the MPS and gate edges
    mps_edge = list(mpslist[index].get_all_dangling())[0]
    gate_edge = gate[0]
    connected = tn.connect(mps_edge, gate_edge)

    # Contract the edge to get the new tensor
    new = tn.contract(connected, name=mpslist[index].name)
    mpslist[index] = new


def apply_one_qubit_gate_to_all(gate: tn.Node, mpslist: List[tn.Node]):
    """Modifies the input mpslist in place by applying a single qubit gate to all tensors in the MPS.

    Args:
        gate: Single qubit gate to apply. A tensor with two free indices.
        mpslist: List of tn.Node objects representing a valid MPS.
    """
    if not is_valid(mpslist):
        raise ValueError("Input mpslist does not define a valid MPS.")

    for i in range(len(mpslist)):
        apply_one_qubit_gate(gate, i, mpslist)


def apply_two_qubit_gate(
        gate: tn.Node,
        indexA: int,
        indexB: int,
        mpslist: List[tn.Node],
        keep_left_canonical: bool = True,
        **kwargs
) -> None:
    """Modifies the input mpslist in place by applying a two qubit gate to the specified nodes.

    Args:
        gate: Two qubit gate to apply.
              Edge convention:
                gate edge 0: Connects to tensor at indexA.
                gate edge 1: Connects to tensor at indexB.
                gate edge 2: Becomes free index of new tensor at indexA after contracting.
                gate edge 3: Becomes free index of new tensor at indexB after contracting.
        indexA: Index of first tensor (qubit) in the mpslist to apply the single qubit gate to.
        indexB: Index of second tensor (qubit) in the mpslist to apply the single qubit gate to.
        mpslist: List of tn.Node objects representing a valid MPS.
        keep_left_canonical: After performing an SVD on the new node to obtain U, S, Vdag,
                             S is grouped with Vdag to form the new right tensor. That is,
                             the left tensor is U, and the right tensor is S @ Vdag. This keeps
                             the MPS in left canonical form if it already was in left canonical form.

                             If False, S is grouped with U so that the new left tensor is U @ S and
                             the new right tensor is Vdag.

    Keyword Arguments:
        max_singular_values (int): Number of singular values to keep.
        max_truncation_err (int): Maximum allowed truncation error by throwing away singular values.
    """
    if not is_valid(mpslist):
        raise ValueError("Input mpslist does not define a valid MPS.")

    if indexA not in range(len(mpslist)) or indexB not in range(len(mpslist)):
        raise ValueError(f"Input tensor indices={(indexA, indexB)} are out of bounds for the input mpslist.")

    if indexA == indexB:
        raise ValueError("Input indices are identical.")

    if abs(indexA - indexB) != 1:
        raise ValueError("Indices must be for adjacent tensors (must differ by one).")

    if len(gate.get_all_dangling()) != 4 or len(gate.get_all_nondangling()) != 0:
        raise ValueError("Two qubit gate must have four free edges and zero connected edges.")

    # Connect the MPS tensors to the gate edges
    if indexA < indexB:
        left_index = indexA
        right_index = indexB
    else:
        raise ValueError(f"IndexA must be less than IndexB.")

    _ = tn.connect(
        list(mpslist[indexA].get_all_dangling())[0], gate.get_edge(0)
    )  # TODO: Which gate edge should be used here?
    _ = tn.connect(
        list(mpslist[indexB].get_all_dangling())[0], gate.get_edge(1)
    )  # TODO: Which gate edge should be used here?

    # Store the free edges of the gate, using the docstring edge convention
    left_gate_edge = gate.get_edge(2)
    right_gate_edge = gate.get_edge(3)

    # Contract the tensors in the MPS
    new_node = tn.contract_between(mpslist[indexA], mpslist[indexB], name="new_mps_tensor")

    # Flatten the two edges from the MPS node to the gate node
    node_gate_edge = tn.flatten_edges_between(new_node, gate)

    # Contract the flattened edge to get a new single MPS node
    new_node = tn.contract(node_gate_edge, name="new_mps_tensor")

    # Get the left and right connected edges (if any)
    left_connected_edge = None
    right_connected_edge = None
    for connected_edge in new_node.get_all_nondangling():
        if "q" in connected_edge.node1.name:
            index = int(connected_edge.node1.name.split("q")[-1])  # Use the "node1" node by default
        else:
            index = int(connected_edge.node2.name.split("q")[-1])  # If "node1" is the new_mps_node, use "node2"

        # Get the connected edges (if any)
        if index <= left_index:
            left_connected_edge = connected_edge
        else:
            right_connected_edge = connected_edge

    # Get the left and right free edges from the original gate
    left_free_edge = left_gate_edge
    right_free_edge = right_gate_edge

    # Group the left (un)connected and right (un)connected edges
    left_edges = [edge for edge in (left_free_edge, left_connected_edge) if edge is not None]
    right_edges = [edge for edge in (right_free_edge, right_connected_edge) if edge is not None]

    # Do the SVD to split the single MPS node into two
    if "max_singular_values" in kwargs.keys():
        maxsvals = kwargs.get("max_singular_values")
        print(f"Truncating SVD, only keeping top {maxsvals} singular value(s).")
    else:
        maxsvals = None

    u, s, vdag, _ = tn.split_node_full_svd(
        new_node,
        left_edges=left_edges,
        right_edges=right_edges,
        max_singular_values=maxsvals,
        left_name="u",
        middle_name="s",
        right_name="vdag"
    )

    # Contract the tensors to keep left or right canonical form
    if keep_left_canonical:
        new_left = u
        new_right = tn.contract_between(s, vdag)
    else:
        new_left = tn.contract_between(u, s)
        new_right = vdag

    # Put the new tensors after applying the gate back into the MPS list
    new_left.name = mpslist[indexA].name
    new_right.name = mpslist[indexB].name

    mpslist[left_index] = new_left
    mpslist[right_index] = new_right
