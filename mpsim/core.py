"""Defines matrix product state class."""

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensornetwork as tn

from mpsim.gates import hgate, rgate, xgate, cnot, swap, is_unitary


class CannotConvertToMPSOperation(Exception):
    pass


class MPSOperation:
    """Defines an operation which an MPS can execute."""
    def __init__(
        self,
        node: tn.Node,
        qudit_indices: Union[int, Tuple[int, ...]],
        qudit_dimension: int = 2
    ) -> None:
        """Initialize an MPS Operation.

        Args:
            node: TensorNetwork node object representing a gate to apply.
                See Notes below.
            qudit_indices: Index/indices of qudits to apply the gate to.
                In an MPS, qudits are indexed from the left starting at zero.
            qudit_dimension: Dimension of qudit(s) to which the MPS Operation
                is applied. Default value is 2 (for qubits).

        Notes:
            The following gate edge conventions describe which edges connect
            to which MPS tensors in one-qudit and two-qudit MPS operations.

            For single qudit gates:
                gate edge 1: Connects to the tensor in the MPS.
                gate edge 0: Becomes the free edge of the new MPS tensor.

            For two qudit gates:

                Let `matrix` be a 4x4 (unitary) matrix which acts on MPS tensors
                at index1 and index2.

                >>> matrix = np.reshape(matrix, newshape=(2, 2, 2, 2))
                >>> gate = tn.Node(matrix)

                ensures the edge convention below is upheld.

                Gate edge convention (assuming index1 < index2)
                    gate edge 2: Connects to tensor at indexA.
                    gate edge 3: Connects to tensor at indexB.
                    gate edge 0: Becomes free index of new tensor at indexA.
                    gate edge 1: Becomes free index of new tensor at indexB.

                If index1 > index2, 0 <--> 1 and 2 <--> 3.
        """
        self._node = node
        if isinstance(qudit_indices, int):
            qudit_indices = (qudit_indices,)
        self._qudit_indices = tuple(qudit_indices)
        self._qudit_dimension = int(qudit_dimension)

    @property
    def qudit_indices(self) -> Tuple[int, ...]:
        """Returns the indices of the qudits that the MPS Operation acts on."""
        return self._qudit_indices

    @property
    def qudit_dimension(self) -> int:
        """Returns the dimension of the qudits the MPS Operation acts on."""
        return self._qudit_dimension

    @property
    def num_qudits(self) -> int:
        """Returns the number of qubits the MPS Operation acts on."""
        return len(self._qudit_indices)

    def node(self, copy: bool = True) -> tn.Node:
        """Returns the node of the MPS Operation.

        Args:
            copy: If True, a copy of the node object is returned.
        """
        if not copy:
            return self._node
        node_dict, _ = tn.copy([self._node])
        return node_dict[self._node]

    def tensor(self, reshape_to_square_matrix: bool = True) -> np.ndarray:
        """Returns a copy of the tensor of the MPS Operation.

        Args:
            reshape_to_square_matrix: If True, the shape of the returned tensor
                    is dim x dim where dim is the qudit dimension raised
                    to the number of qudits that the MPS Operator acts on.
        """
        tensor = deepcopy(self._node.tensor)
        if reshape_to_square_matrix:
            dim = self._qudit_dimension ** self.num_qudits
            tensor = np.reshape(
                tensor, newshape=(dim, dim)
            )
        return tensor

    def is_valid(self) -> bool:
        """Returns True if the MPS Operation is valid, else False.

        A valid MPS Operation meets the following criteria:
            (1) Tensor of gate has shape (d, ..., d) where d is the qudit
                dimension and there are 2 * num_qudits entries in the tuple.
            (2) Tensor has 2n free edges where n = number of qudits.
            (3) All tensor edges are free edges.
        """
        d = self._qudit_dimension
        if not self._node.tensor.shape == tuple([d] * 2 * self.num_qudits):
            return False
        if not len(self._node.get_all_edges()) == 2 * self.num_qudits:
            return False
        if self._node.has_nondangling_edge():
            return False
        return True

    def is_unitary(self) -> bool:
        """Returns True if the MPS Operation is unitary, else False.

        An MPS Operation is unitary if its gate tensor U is unitary, i.e. if
        U^dag @ U = U @ U^dag = I.
        """
        return is_unitary(self.tensor())

    def is_single_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on a single qudit."""
        return self.num_qudits == 1

    def is_two_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on two qudits."""
        return self.num_qudits == 2

    def __str__(self):
        return f"Tensor {self._node.name} on qudit(s) {self._qudit_indices}."


class MPS:
    """Matrix Product State (MPS) for simulating quantum circuits."""
    def __init__(
            self,
            nqudits: int,
            qudit_dimension: int = 2,
            tensor_prefix: str = "q"
    ) -> None:
        """Initializes an MPS of qudits in the all |0> state.

        The MPS has the following structure (shown for six qudits):

            @ ---- @ ---- @ ---- @ ---- @ ---- @
            |      |      |      |      |      |

        Virtual indices have bond dimension one and physical indices
        have bond dimension equal to the qudit_dimension.

        Args:
            nqudits: Number of qubits in the all zero state.
            qudit_dimension: Dimension of qudits. Default value is 2 (qubits).
            tensor_prefix: Prefix for tensor names.
                The full name is prefix + numerical index, numbered from
                left to right starting with zero.
        """
        if nqudits < 2:
            raise ValueError(
                f"Number of qudits must be greater than 2 but is {nqudits}."
            )

        # Set nodes on the interior
        nodes = [
            tn.Node(
                np.array(
                    [[[1.0]], *[[[0]]] * (qudit_dimension - 1)],
                    dtype=np.complex64
                ),
                name=tensor_prefix + str(x + 1),
            )
            for x in range(nqudits - 2)
        ]

        # Set nodes on the left and right edges
        nodes.insert(
            0,
            tn.Node(
                np.array(
                    [[1.0], *[[0]] * (qudit_dimension - 1)], dtype=np.complex64
                ),
                name=tensor_prefix + str(0),
            ),
        )
        nodes.append(
            tn.Node(
                np.array(
                    [[1.0], *[[0]] * (qudit_dimension - 1)], dtype=np.complex64
                ),
                name=tensor_prefix + str(nqudits - 1),
            )
        )

        # Connect edges between interior nodes
        for i in range(1, nqudits - 2):
            tn.connect(nodes[i].get_edge(2), nodes[i + 1].get_edge(1))

        # Connect edge nodes to their neighbors
        if nqudits < 3:
            tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
        else:
            tn.connect(nodes[0].get_edge(1), nodes[1].get_edge(1))
            tn.connect(nodes[-1].get_edge(1), nodes[-2].get_edge(2))

        self._nqudits = nqudits
        self._qudit_dimension = qudit_dimension
        self._prefix = tensor_prefix
        self._nodes = nodes
        self._max_bond_dimensions = [
            self._qudit_dimension ** (i + 1) for i in range(self._nqudits // 2)
        ]
        self._max_bond_dimensions += list(reversed(self._max_bond_dimensions))
        if self._nqudits % 2 == 0:
            self._max_bond_dimensions.remove(
                self._qudit_dimension ** (self._nqudits // 2)
            )
        self._fidelities = []  # type: List[float]

    @staticmethod
    def from_wavefunction(
        wavefunction: np.ndarray,
        nqudits: int,
        qudit_dimension: int = 2,
        tensor_prefix: str = "q"
    ) -> 'MPS':
        """Returns an MPS constructed from the initial wavefunction.

        Args:
            wavefunction: Vector (numpy array) representing the wavefunction.
            nqudits: Number of qudits in the wavefunction.
            qudit_dimension: Dimension of qudits. (Default is 2 for qubits.)
            tensor_prefix: Prefix for tensor names.
                The full name is prefix + numerical index, numbered from
                left to right starting with zero.

        Raises:
            TypeError: Wavefunction is not a numpy.ndarray or cannot be
                converted to one.
            ValueError: If:
                * The wavefunction is not one-dimensional (a vector).
                * If the number of elements in the wavefunction is not
                  equal to qudit_dimension ** nqudits.
                * nqudits is less than two.
        """
        if not isinstance(wavefunction, (list, tuple, np.ndarray)):
            raise TypeError("Invalid type for wavefunction.")
        wavefunction = np.array(wavefunction)

        if len(wavefunction.shape) != 1:
            raise ValueError(
                "Invalid shape for wavefunction. Should be a vector."
            )

        if nqudits < 2:
            raise ValueError("At least two qudits are required.")

        if wavefunction.size != qudit_dimension ** nqudits:
            raise ValueError(
                "Mismatch between wavefunction, qudit_dimension, and nqudits. "
                f"Expected {qudit_dimension ** nqudits} elements in the "
                f"wavefunction, but wavefunction has {wavefunction.size} "
                f"elements."
            )

        # Reshape the wavefunction
        wavefunction = np.reshape(
            wavefunction, newshape=[qudit_dimension] * nqudits
        )
        to_split = tn.Node(
            wavefunction, axis_names=[str(i) for i in range(nqudits)]
        )

        # Perform SVD across each cut
        # TODO: There must be a better way of indexing edges...
        nodes = []
        for i in range(nqudits - 1):
            left_edges = []
            right_edges = []
            for edge in to_split.get_all_dangling():
                if edge.name == str(i):
                    left_edges.append(edge)
                else:
                    right_edges.append(edge)
            if nodes:
                for edge in nodes[-1].get_all_nondangling():
                    if to_split in edge.get_nodes():
                        left_edges.append(edge)

            left_node, right_node, _ = tn.split_node(
                to_split,
                left_edges,
                right_edges,
                left_name=tensor_prefix + str(i)
            )
            nodes.append(left_node)
            to_split = right_node
        to_split.name = tensor_prefix + str(nqudits - 1)
        nodes.append(to_split)

        # Return the MPS
        mps = MPS(nqudits, qudit_dimension, tensor_prefix)
        mps._nodes = nodes
        return mps

    @property
    def nqudits(self) -> int:
        """Returns the number of qudits in the MPS."""
        return self._nqudits

    @property
    def qudit_dimension(self) -> int:
        """Returns the dimension of each qudit in the MPS."""
        return self._qudit_dimension

    def get_bond_dimension_of(self, index: int) -> int:
        """Returns the bond dimension of the right edge of the node
        at the given index.

        Args:
            index: Index of the node.
                The returned bond dimension is that of the right edge
                of the given node.
        """
        if not self.is_valid():
            raise ValueError("MPS is invalid.")

        if index >= self._nqudits:
            raise ValueError(
                f"Index should be less than {self._nqudits} but is {index}."
            )

        left = self.get_node(index, copy=False)
        right = self.get_node(index + 1, copy=False)
        tn.check_connected((left, right))
        edge = tn.get_shared_edges(left, right).pop()
        return edge.dimension

    def get_bond_dimensions(self) -> List[int]:
        """Returns the bond dimensions of the MPS."""
        return [self.get_bond_dimension_of(i) for i in range(self._nqudits - 1)]

    def get_max_bond_dimension_of(self, index: int) -> int:
        """Returns the maximum bond dimension of the right edge
        of the node at the given index.

        Args:
            index: Index of the node.
                The returned bond dimension is that of the right edge
                of the given node.
                Negative indices count backwards from the right of the MPS.
        """
        if index >= self._nqudits:
            raise ValueError(
                f"Index should be less than {self._nqudits} but is {index}."
            )
        return self._max_bond_dimensions[index]

    def get_max_bond_dimensions(self) -> List[int]:
        """Returns the maximum bond dimensions of the MPS."""
        return self._max_bond_dimensions

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
            copy: If true, a copy of the node is returned,
                   else the actual node is returned.
        """
        return self.get_nodes(copy)[i]

    def get_free_edge_of(self, index: int, copy: bool = True) -> tn.Edge:
        """Returns the free (dangling) edge of a node with specified index.
        
        Args:
            index: Specifies the node.
            copy: If True, returns a copy of the edge.
                  If False, returns the actual edge.
        """
        return self.get_node(index, copy).get_all_dangling().pop()

    def get_left_connected_edge_of(self, index: int) -> Union[tn.Edge, None]:
        """Returns the left connected edge of the specified node, if it exists.

        Args:
            index: Index of node to get the left connected edge of.
        """
        if index == 0:
            return None

        return tn.get_shared_edges(
            self._nodes[index], self._nodes[index - 1]
        ).pop()

    def get_right_connected_edge_of(self, index: int) -> Union[tn.Edge, None]:
        """Returns the left connected edge of the specified node, if it exists.

        Args:
            index: Index of node to get the right connected edge of.
        """
        if index == self._nqudits - 1:
            return None

        return tn.get_shared_edges(
            self._nodes[index], self._nodes[index + 1]
        ).pop()

    @property
    def wavefunction(self) -> np.array:
        """Returns the wavefunction of the MPS as a vector."""
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

        return np.reshape(
            fin.tensor, newshape=(self._qudit_dimension ** self._nqudits)
        )

    def norm(self) -> float:
        """Returns the norm of the MPS computed by contraction."""
        a = self.get_nodes(copy=True)
        b = self.get_nodes(copy=True)
        for n in b:
            n.set_tensor(np.conj(n.tensor))

        for i in range(self._nqudits):
            tn.connect(
                a[i].get_all_dangling().pop(), b[i].get_all_dangling().pop()
            )

        for i in range(self._nqudits - 1):
            # TODO: Optimize by flattening edges
            mid = tn.contract_between(a[i], b[i])
            new = tn.contract_between(mid, a[i + 1])
            a[i + 1] = new

        fin = tn.contract_between(a[-1], b[-1])
        assert len(fin.edges) == 0  # Debug check
        assert np.isclose(np.imag(fin.tensor), 0.0)  # Debug check
        return abs(fin.tensor)

    def renormalize(self, to_norm: float = 1.0) -> None:
        """Renormalizes the MPS.

        Args:
            to_norm: The resulting MPS will have this norm.

        Raises:
            ValueError: If to_norm is negative or too close to zero, or if
                        the current norm of the MPS is too close to zero.
        """
        if to_norm <= 0.:
            raise ValueError(f"Arg to_norm must be positive but is {to_norm}")

        if np.isclose(to_norm, 0., atol=1e-15):
            raise ValueError(
                f"Arg to_norm = {to_norm} is too close to numerical zero."
            )

        if np.isclose(self.norm(), 0., atol=1e-15):
            raise ValueError(
                "Norm of MPS is numerically zero, cannot renormalize."
            )

        norm = self.norm()
        for i, node in enumerate(self._nodes):
            self._nodes[i].set_tensor(
                np.sqrt(to_norm / norm)**(1 / self.nqudits) * node.tensor
            )

    def apply_one_qubit_gate(
            self,
            gate: tn.Node,
            index: int,
            ortho_after_non_unitary: bool = True,
            renormalize_after_non_unitary: bool = True,
    ) -> None:
        """Applies a single qubit gate to a specified node.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
            index: Index of tensor (qubit) in the MPS to apply
                the single qubit gate to.
            ortho_after_non_unitary: If True, orthonormalizes edge(s) of the
                node after applying a non-unitary gate.
            renormalize_after_non_unitary: If True, renormalize the MPS after
                applying a non-unitary gate.

        Raises:
            ValueError: On invalid MPS, invalid index, or invalid gate.
        """
        if not self.is_valid():
            raise ValueError("MPS is invalid.")

        if index not in range(self._nqudits):
            raise ValueError(
                f"Input tensor index={index} is out of bounds for"
                f" an MPS on {self._nqudits} qubits."
            )

        if (len(gate.get_all_dangling()) != 2
                or len(gate.get_all_nondangling()) != 0):
            raise ValueError(
                "Single qubit gate must have two free edges"
                " and zero connected edges."
            )

        # TODO: Check that the edge dimension of the gate matches the MPS edge
        #  dimension.

        # Store the norm for optional renormalization after non-unitary gate
        if not is_unitary(gate) and renormalize_after_non_unitary:
            norm = self.norm()

        # Connect the MPS and gate edges
        mps_edge = list(self._nodes[index].get_all_dangling())[0]
        gate_edge = gate[1]  # TODO: Is this the correct edge to use (always)?
        connected = tn.connect(mps_edge, gate_edge)

        # Contract the edge to get the new tensor
        new = tn.contract(connected, name=self._nodes[index].name)
        self._nodes[index] = new

        # Optional orthonormalization after a non-unitary gate
        # TODO: Allow for setting a different threshold in ortho funcs here.
        if not is_unitary(gate) and ortho_after_non_unitary:
            # Edge case: Left-most node
            if index == 0:
                self.orthonormalize_right_edge_of(index)

            # Edge case: Right-most node
            elif index == self._nqudits - 1:
                self.orthonormalize_left_edge_of(index)

            # General case
            else:
                self.orthonormalize_right_edge_of(index)
                self.orthonormalize_left_edge_of(index)

        # Optional renormalization after non-unitary gate
        if not is_unitary(gate) and renormalize_after_non_unitary:
            self.renormalize(norm)

    def orthonormalize_right_edge_of(
            self, node_index: int, threshold: float = 1e-8
    ) -> None:
        """Performs SVD on the specified node to orthonormalize the right edge.

        Let N be the specified node. Then, this function:
            (1) Performs SVD on N to get N = U @ S @ Vdag,
            (2) Sets the node at N to be U, and
            (3) Sets the node to the right of N (call it M) to be S @ Vdag @ M.

        Args:
            node_index: Index which specifies the node.
            threshold: Throw away singular values below threshold * self.norm().

        Raises:
            ValueError: If the node_index is out of bounds for the MPS.
        """
        if not 0 <= node_index < self._nqudits - 1:
            raise ValueError("Invalid edge index.")

        # Get the node
        node = self._nodes[node_index]

        # Get the left and right edges to do the SVD
        left_edges = [self.get_free_edge_of(node_index, copy=False)]
        if self.get_left_connected_edge_of(node_index):
            left_edges.append(self.get_left_connected_edge_of(node_index))
        right_edges = [self.get_right_connected_edge_of(node_index)]

        # Do the SVD
        u, s, vdag, _ = tn.split_node_full_svd(
            node,
            left_edges=left_edges,
            right_edges=right_edges,
            max_truncation_err=threshold * self.norm(),
        )

        # Set the new node
        self._nodes[node_index] = u
        self._nodes[node_index].name = self._prefix + str(node_index)

        # Mutlipy S and Vdag to the right
        temp = tn.contract_between(s, vdag)
        new_right = tn.contract_between(temp, self._nodes[node_index + 1])
        new_right.name = self._nodes[node_index + 1].name
        self._nodes[node_index + 1] = new_right

    def orthonormalize_left_edge_of(
            self, node_index: int, threshold: float = 1e-8
    ) -> None:
        """Performs SVD on the specified node to orthonormalize the left edge.

        Let M be the specified node. Then, this function:
            (1) Performs SVD on M to get M = U @ S @ Vdag,
            (2) Sets the node at M to be Vdag, and
            (3) Sets the node to the left of M (call it N) to be N @ U @ S.

        Args:
            node_index: Index which specifies the node.
            threshold: Throw away singular values below threshold * self.norm().

        Raises:
            ValueError: If the node_index is out of bounds for the MPS.
        """
        if not 0 < node_index <= self._nqudits - 1:
            raise ValueError("Invalid edge index.")

        # Get the node
        node = self._nodes[node_index]

        # Get the left and right edges to do the SVD
        left_edges = [self.get_left_connected_edge_of(node_index)]
        right_edges = [self.get_free_edge_of(node_index, copy=False)]
        if self.get_right_connected_edge_of(node_index):
            right_edges.append(self.get_right_connected_edge_of(node_index))

        # Do the SVD
        u, s, vdag, _ = tn.split_node_full_svd(
            node,
            left_edges=left_edges,
            right_edges=right_edges,
            max_truncation_err=threshold * self.norm(),
        )

        # Set the new node
        self._nodes[node_index] = vdag
        self._nodes[node_index].name = self._prefix + str(node_index)

        # Mutlipy U and S to the left
        temp = tn.contract_between(u, s)
        new_left = tn.contract_between(self._nodes[node_index - 1], temp)
        new_left.name = self._nodes[node_index - 1].name
        self._nodes[node_index - 1] = new_left

    def apply_one_qubit_gate_to_all(self, gate: tn.Node) -> None:
        """Applies a single qubit gate to all tensors in the MPS.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
        """
        for i in range(self._nqudits):
            self.apply_one_qubit_gate(gate, i)

    def move_node_from_left_to_right(
            self, current_node_index: int, final_node_index: int, **kwargs
    ) -> None:
        """Moves the MPS node at current_node_index to the final_node_index by
        implementing a sequence of SWAP gates from left to right.
        
        Args:
            current_node_index: Index of the node to move from left to right.
            final_node_index: Final index location of the node to move.
        """
        if current_node_index > final_node_index:
            raise ValueError(
                "current_node_index should be smaller than final_node_index."
            )

        if current_node_index < 0:
            raise ValueError("current_node_index out of range.")

        if final_node_index >= self._nqudits:
            raise ValueError("final_node_index out of range.")

        if current_node_index == final_node_index:
            return

        while current_node_index < final_node_index:
            self.swap(current_node_index, current_node_index + 1, **kwargs)
            current_node_index += 1

    def move_node_from_right_to_left(
            self, current_node_index: int, final_node_index: int, **kwargs
    ) -> None:
        """Moves the MPS node at current_node_index to the final_node_index by
        implementing a sequence of SWAP gates from right to left.

        Args:
            current_node_index: Index of the node to move from right to left.
            final_node_index: Final index location of the node to move.
        """
        if current_node_index < final_node_index:
            raise ValueError(
                "current_node_index should be larger than final_node_index."
            )

        if current_node_index > self._nqudits:
            raise ValueError("current_node_index out of range.")

        if final_node_index < 0:
            raise ValueError("final_node_index out of range.")

        if current_node_index == final_node_index:
            return

        while current_node_index > final_node_index:
            self.swap(current_node_index - 1, current_node_index, **kwargs)
            current_node_index -= 1

    def apply_two_qubit_gate(
        self, gate: tn.Node, indexA: int, indexB: int, **kwargs
    ) -> None:
        """Applies a two qubit gate to the specified nodes.

        Args:
            gate: Two qubit gate to apply. See Notes for the edge convention.
            indexA: Index of first tensor (qubit) in the mpslist to apply the
                     single qubit gate to.
            indexB: Index of second tensor (qubit) in the mpslist to apply the
                     single qubit gate to.

        Keyword Arguments:
            keep_left_canonical: After performing an SVD on the new node to
                obtain U, S, Vdag, S is grouped with Vdag to form the
                new right tensor. That is, the left tensor is U, and the
                right tensor is S @ Vdag. This keeps the MPS in left canonical
                form if it already was in left canonical form.

                If False, S is grouped with U so that the new left tensor
                 is U @ S and the new right tensor is Vdag.

            maxsvals (int): Number of singular values to keep
                for all two-qubit gates.

            fraction (float): Number of singular values to keep expressed as a
                fraction of the maximum bond dimension.
                Must be between 0 and 1, inclusive.

        Notes:
            The following gate edge convention is used to connect gate edges to
            MPS edges. Let `matrix` be a 4x4 (unitary) matrix. Then,

            >>> matrix = np.reshape(matrix, newshape=(2, 2, 2, 2))
            >>> gate = tn.Node(matrix)

            ensures the edge convention below is upheld.

            Gate edge convention (assuming indexA < indexB)
                gate edge 2: Connects to tensor at indexA.
                gate edge 3: Connects to tensor at indexB.
                gate edge 0: Becomes free index of new tensor at indexA.
                gate edge 1: Becomes free index of new tensor at indexB.

            If indexA > indexB, 0 <--> 1 and 2 <--> 3.
        """
        if not self.is_valid():
            raise ValueError("MPS is not valid.")

        if (indexA not in range(self._nqudits)
                or indexB not in range(self.nqudits)):
            raise ValueError(
                f"Input tensor indices={(indexA, indexB)} are out of bounds"
                f" for an MPS on {self._nqudits} qubits."
            )

        if indexA == indexB:
            raise ValueError("Input indices cannot be identical.")

        if (len(gate.get_all_dangling()) != 4
                or len(gate.get_all_nondangling()) != 0):
            raise ValueError(
                "Two qubit gate must have four free edges"
                " and zero connected edges."
            )

        # Flip the "control"/"target" gate edges and tensor edges if needed
        if indexB < indexA:
            gate.reorder_edges([gate[1], gate[0], gate[3], gate[2]])
            indexA, indexB = indexB, indexA

        # Swap tensors until adjacent if necessary
        invert_swap_network = False
        if indexA < indexB - 1:
            invert_swap_network = True
            original_indexA = indexA
            self.move_node_from_left_to_right(indexA, indexB - 1, **kwargs)
            indexA = indexB - 1

        # Connect the MPS tensors to the gate edges
        left_index = indexA
        right_index = indexB

        _ = tn.connect(
            self.get_free_edge_of(index=indexA, copy=False),
            gate.get_edge(2)
        )
        _ = tn.connect(
            self.get_free_edge_of(index=indexB, copy=False),
            gate.get_edge(3)
        )

        # Store the free edges of the gate
        left_gate_edge = gate.get_edge(0)
        right_gate_edge = gate.get_edge(1)

        # Contract the tensors in the MPS
        new_node = tn.contract_between(
            self._nodes[indexA], self._nodes[indexB], name="new_mps_tensor"
        )

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
        left_edges = [
            edge for edge in (left_free_edge, left_connected_edge)
            if edge is not None
        ]
        right_edges = [
            edge for edge in (right_free_edge, right_connected_edge)
            if edge is not None
        ]

        # ================================================
        # Do the SVD to split the single MPS node into two
        # ================================================
        # Options for canonicalization + truncation
        if "keep_left_canonical" in kwargs.keys():
            keep_left_canonical = kwargs.get("keep_left_canonical")
        else:
            keep_left_canonical = True

        if "fraction" in kwargs.keys() and "maxsvals" in kwargs.keys():
            raise ValueError(
                "Only one of (fraction, maxsvals) can be provided as kwargs."
            )

        if "fraction" in kwargs.keys():
            fraction = kwargs.get("fraction")
            if not (0 <= fraction <= 1):
                raise ValueError(
                    "Keyword fraction must be between 0 and 1 but is", fraction
                )
            maxsvals = int(
                round(fraction * self.get_max_bond_dimension_of(
                    min(indexA, indexB)
                ))
            )
        else:
            maxsvals = None  # Keeps all singular values

        if "maxsvals" in kwargs.keys():
            maxsvals = int(kwargs.get("maxsvals"))

        u, s, vdag, truncated_svals = tn.split_node_full_svd(
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

        # Invert the Swap network, if necessary
        if invert_swap_network:
            self.move_node_from_right_to_left(indexA, original_indexA, **kwargs)

        # TODO: Remove. This is only for convenience in benchmarking.
        self._fidelities.append(self.norm())

    # TODO: Take single qudit gate application kwargs/options into account
    def apply_mps_operation(
            self, mps_operation: MPSOperation, **kwargs
    ) -> None:
        """Applies the MPS Operation to the MPS.

        Args:
            mps_operation: Valid MPS Operation to apply to the MPS.

        Keyword Args:
            See MPS.apply_two_qubit_gate.
        """
        if not mps_operation.is_valid():
            raise ValueError("Input MPS Operation is not valid.")

        if mps_operation.is_single_qudit_operation():
            self.apply_one_qubit_gate(
                mps_operation.node(), *mps_operation.qudit_indices
            )
        elif mps_operation.is_two_qudit_operation():
            self.apply_two_qubit_gate(
                mps_operation.node(), *mps_operation.qudit_indices, **kwargs
            )
        else:
            raise ValueError(
                "Only one-qudit and two-qudit gates are currently supported."
            )

    def apply_mps_operations(
            self, mps_operations: Sequence[MPSOperation], **kwargs
    ):
        """Applies the sequence of MPS Operations to the MPS.

        Args:
            mps_operations: List of valid MPS Operations to apply to the MPS.

        Keyword Args:
            See MPS.apply_two_qubit_gate.
        """
        for mps_operation in mps_operations:
            self.apply_mps_operation(mps_operation, **kwargs)

    # TODO: Remove single qubit gates -- these don't generalize to qudits.
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

    def r(self, index, seed: Optional[int] = None,
          angle_scale: float = 1.0) -> None:
        """Applies a random rotation to the qubit indexed by `index`.

        If index == -1, (different) random rotations are applied to all qubits.
        Args:
            index: Index of tensor to apply rotation to.
            seed: Seed for random number generator.
            angle_scale: Floating point value to scale angles by. Default 1.
        """
        if index == -1:
            for i in range(self._nqudits):
                self.apply_one_qubit_gate(
                    rgate(seed, angle_scale), i,
                )
        else:
            self.apply_one_qubit_gate(rgate(seed, angle_scale), index)

    # TODO: Remove. This doesn't generalize to qudits.
    def cnot(self, a: int, b: int, **kwargs) -> None:
        """Applies a CNOT gate with qubit indexed `a` as control
        and qubit indexed `b` as target.
        """
        self.apply_two_qubit_gate(cnot(), a, b, **kwargs)

    def sweep_cnots_left_to_right(self, **kwargs) -> None:
        """Applies a layer of CNOTs between adjacent qubits
        going from left to right.
        """
        for i in range(0, self._nqudits - 1, 2):
            self.cnot(i, i + 1, keep_left_canonical=True, **kwargs)

    def sweep_cnots_right_to_left(self, **kwargs) -> None:
        """Applies a layer of CNOTs between adjacent qubits
         going from right to left.
         """
        for i in range(self._nqudits - 2, 0, -2):
            self.cnot(i - 1, i, keep_left_canonical=False, **kwargs)

    def swap(self, a: int, b: int, **kwargs) -> None:
        """Applies a SWAP gate between qubits indexed `a` and `b`."""
        if b < a:
            a, b = b, a
        self.apply_two_qubit_gate(swap(), a, b, **kwargs)

    def __str__(self):
        return "----".join(str(tensor) for tensor in self._nodes)
