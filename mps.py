"""Code for Matrix Product State initialization, manipulation, and operations."""

from copy import deepcopy
from typing import (List, Optional)

import numpy as np
from scipy.linalg import expm
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
def igate() -> tn.Node:
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


def rgate(seed: Optional[int] = None):
    """Returns the random single qubit gate described in https://arxiv.org/abs/2002.07730."""
    if seed:
        np.random.seed(seed)

    # Get the random parameters
    theta, alpha, phi = np.random.rand(3) * 2 * np.pi
    mx = np.sin(alpha) * np.cos(phi)
    my = np.sin(alpha) * np.sin(phi)
    mz = np.cos(alpha)

    # Get the unitary
    unitary = expm(-1j * theta * (mx * _xmatrix + my * _ymatrix * mz * _zmatrix))

    # TODO: Note to Guifre diagonal elements of this unitary are always real, and off-diagonal elements are imaginary
    return tn.Node(unitary)


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


class MPS:
    """Matrix Product State (MPS) for simulating (noisy) quantum circuits."""
    def __init__(self, nqubits: int, tensor_prefix: str = "q") -> None:
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

        self._nqubits = nqubits
        self._prefix = tensor_prefix
        self._nodes = nodes

    @property
    def nqubits(self):
        return self._nqubits

    def is_valid(self) -> bool:
        """Returns true if the mpslist defines a valid MPS, else False.

        A valid MPS satisfies the following criteria:
            (1) At least two tensors.
            (2) Every tensor has exactly one free (dangling) edge.
            (3) Every tensor has connected edges to its nearest neighbor(s).
        """
        if len(self._nodes) < 2:
            return False

        for (i, tensor) in enumerate(self._nodes):
            # Exterior nodes
            if i == 0 or i == len(self._nodes) - 1:
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

            if i < len(self._nodes) - 1:
                try:
                    tn.check_connected((self._nodes[i], self._nodes[i + 1]))
                except ValueError:
                    print(f"Nodes at index {i} and {i + 1} are not connected.")
                    return False
        return True

    def get_nodes(self, copy: bool = True) -> List[tn.Node]:
        """Returns """
        if not copy:
            return self._nodes
        nodes_dict, _ = tn.copy(self._nodes)
        return list(nodes_dict.values())

    def get_node(self, i: int, copy: bool = True) -> tn.Node:
        """Returns the ith node in the MPS counting from the left.

        Args:
            i: Index of node to get.
            copy: If true, a copy of the node is returned, else the actual node is returned.
        """
        return self.get_nodes(copy)[i]

    @property
    def wavefunction(self) -> np.array:
        """Returns the wavefunction of a valid MPS as a (giant) vector by contracting all virtual indices."""
        if not self.is_valid():
            raise ValueError("MPS is not valid.")

        # Replicate the mps
        nodes = self.get_nodes(copy=True)
        fin = nodes.pop(0)
        for node in nodes:
            fin = tn.contract_between(fin, node)

        # Make sure all edges are free
        if set(fin.get_all_dangling()) != set(fin.get_all_edges()):
            raise ValueError("Invalid MPS.")

        return np.reshape(fin.tensor, newshape=(2**self._nqubits))

    def apply_one_qubit_gate(self, gate: tn.Node, index: int) -> None:
        """Modifies the input mpslist in place by applying a single qubit gate to a specified node.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
            index: Index of tensor (qubit) in the mpslist to apply the single qubit gate to.
        """
        if not self.is_valid():
            raise ValueError("Input mpslist does not define a valid MPS.")

        if index not in range(self._nqubits):
            raise ValueError(f"Input tensor index={index} is out of bounds for the input mpslist.")

        if len(gate.get_all_dangling()) != 2 or len(gate.get_all_nondangling()) != 0:
            raise ValueError("Single qubit gate must have two free edges and zero connected edges.")

        # Connect the MPS and gate edges
        mps_edge = list(self._nodes[index].get_all_dangling())[0]
        gate_edge = gate[0]
        connected = tn.connect(mps_edge, gate_edge)

        # Contract the edge to get the new tensor
        new = tn.contract(connected, name=self._nodes[index].name)
        self._nodes[index] = new

    def apply_one_qubit_gate_to_all(self, gate: tn.Node) -> None:
        """Modifies the input mpslist in place by applying a single qubit gate to a specified node.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
        """
        for i in range(self._nqubits):
            self.apply_one_qubit_gate(gate, i)

    def apply_two_qubit_gate(
            self,
            gate: tn.Node,
            indexA: int,
            indexB: int,
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

        Keyword Arguments:
            keep_left_canonical: After performing an SVD on the new node to obtain U, S, Vdag,
                                 S is grouped with Vdag to form the new right tensor. That is,
                                 the left tensor is U, and the right tensor is S @ Vdag. This keeps
                                 the MPS in left canonical form if it already was in left canonical form.

                                 If False, S is grouped with U so that the new left tensor is U @ S and
                                 the new right tensor is Vdag.
            max_singular_values (int): Number of singular values to keep.
            max_truncation_err (int): Maximum allowed truncation error by throwing away singular values.
        """
        if not self.is_valid():
            raise ValueError("Input mpslist does not define a valid MPS.")

        if indexA not in range(self._nqubits) or indexB not in range(self.nqubits):
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
            list(self._nodes[indexA].get_all_dangling())[0], gate.get_edge(0)
        )  # TODO: Which gate edge should be used here?
        _ = tn.connect(
            list(self._nodes[indexB].get_all_dangling())[0], gate.get_edge(1)
        )  # TODO: Which gate edge should be used here?

        # Store the free edges of the gate, using the docstring edge convention
        left_gate_edge = gate.get_edge(2)
        right_gate_edge = gate.get_edge(3)

        # Contract the tensors in the MPS
        new_node = tn.contract_between(self._nodes[indexA], self._nodes[indexB], name="new_mps_tensor")

        # Flatten the two edges from the MPS node to the gate node
        node_gate_edge = tn.flatten_edges_between(new_node, gate)

        # Contract the flattened edge to get a new single MPS node
        new_node = tn.contract(node_gate_edge, name="new_mps_tensor")

        # Get the left and right connected edges (if any)
        left_connected_edge = None
        right_connected_edge = None
        for connected_edge in new_node.get_all_nondangling():
            if self._prefix in connected_edge.node1.name:
                # Use the "node1" node by default
                index = int(connected_edge.node1.name.split(self._prefix)[-1])
            else:
                # If "node1" is the new_mps_node, use "node2"
                index = int(connected_edge.node2.name.split(self._prefix)[-1])

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

        # ================================================
        # Do the SVD to split the single MPS node into two
        # ================================================
        # Options for canonicalization + truncation
        if "keep_left_canonical" in kwargs.keys():
            keep_left_canonical = kwargs.get("keep_left_canonical")
        else:
            keep_left_canonical = True
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
        )

        # Contract the tensors to keep left or right canonical form
        if keep_left_canonical:
            new_left = u
            new_right = tn.contract_between(s, vdag)
        else:
            new_left = tn.contract_between(u, s)
            new_right = vdag

        # Put the new tensors after applying the gate back into the MPS list
        new_left.name = self._nodes[indexA].name
        new_right.name = self._nodes[indexB].name

        self._nodes[left_index] = new_left
        self._nodes[right_index] = new_right

    def x(self, index: int) -> None:
        """Applies a NOT (Pauli-X) gate to a qubit specified by the index.

        If index == -1, the gate is applied to all qubits.

        Args:
            index: Index of qubit (tensor) to apply X gate to.
        """
        if index == -1:
            self.apply_one_qubit_gate_to_all(xgate())
        else:
            self.apply_one_qubit_gate(xgate(), index)

    def h(self, index: int) -> None:
        """Applies a Hadamard gate to a qubit specified by the index.

        If index == -1, the gate is applied to all qubits.

        Args:
            index: Index of qubit (tensor) to apply X gate to.
        """
        if index == -1:
            self.apply_one_qubit_gate_to_all(hgate())
        else:
            self.apply_one_qubit_gate(hgate(), index)

    def r(self, index, seed: Optional[int] = None) -> None:
        """Applies a random rotation to the qubit indexed by `index`.

        If index == -1, random rotations are applied to all qubits. (Different rotations.)

        """
        if index == -1:
            for i in range(self._nqubits):
                self.apply_one_qubit_gate(rgate(seed), i)
        else:
            self.apply_one_qubit_gate(rgate(seed), index)

    def cnot(self, a: int, b: int, **kwargs) -> None:
        """Applies a CNOT gate with qubit indexed `a` as control."""
        self.apply_two_qubit_gate(cnot(), a, b, **kwargs)

    def sweep_cnots_left_to_right(self, keep: int) -> None:
        """Applies a layer of CNOTs between adjacent qubits going from left to right."""
        for i in range(0, self._nqubits - 1, 2):
            self.cnot(i, i + 1, keep_left_canonical=True, max_singular_values=keep)

    def sweep_cnots_right_to_left(self, keep: int) -> None:
        """Applies a layer of CNOTs between adjacent qubits going from right to left."""
        for i in range(self._nqubits - 2, 0, -2):
            self.cnot(i - 1, i, keep_left_canonical=False, max_singular_values=keep)

    def swap(self, a: int, b: int, **kwargs) -> None:
        """Applies a SWAP gate between qubits indexed `a` and `b`."""
        self.apply_two_qubit_gate(swap(), a, b, **kwargs)
