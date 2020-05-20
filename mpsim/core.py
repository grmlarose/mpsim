"""Defines matrix product state class."""

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import tensornetwork as tn

from mpsim.gates import (
    hgate, rgate, xgate, cnot, swap, is_unitary, is_hermitian,
    computational_basis_state
)


class CannotConvertToMPSOperation(Exception):
    pass


class MPSOperation:
    """Defines an operation which can act on a matrix product state."""
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
        return is_unitary(self.tensor(reshape_to_square_matrix=True))

    def is_hermitian(self) -> bool:
        """Returns True if the MPS Operation is Hermitian, else False.

        An MPS Operation is Hermitian if its gate tensor M is Hermitian, i.e.
        if M^dag = M.
        """
        return is_hermitian(self.tensor(reshape_to_square_matrix=True))

    def is_single_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on a single qudit."""
        return self.num_qudits == 1

    def is_two_qudit_operation(self) -> bool:
        """Returns True if the MPS Operation acts on two qudits."""
        return self.num_qudits == 2

    def __str__(self):
        return f"Tensor {self._node.name} on qudit(s) {self._qudit_indices}."


class MPS:
    """Matrix Product State (MPS) object."""
    def __init__(
        self,
        nqudits: int,
        qudit_dimension: int = 2,
        tensor_prefix: str = "q"
    ) -> None:
        """Initializes an MPS of qudits in the ground (all-zero) state.

        The MPS has the following structure (shown for six qudits):

            @ ---- @ ---- @ ---- @ ---- @ ---- @
            |      |      |      |      |      |

        Virtual indices have bond dimension one (initially) and physical indices
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
                name=tensor_prefix + str(i + 1),
            )
            for i in range(nqudits - 2)
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
        self._norms = []  # type: List[float]

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

    def bond_dimension_of(self, node_index: int) -> int:
        """Returns the bond dimension of the right edge of the node
        at the given index.

        Args:
            node_index: Index of the node.
                The returned bond dimension is that of the right edge
                of the given node.
        """
        if not self.is_valid():
            raise ValueError("MPS is invalid.")

        if node_index >= self._nqudits:
            raise ValueError(
                f"Index should be less than {self._nqudits} but is {node_index}."
            )

        left = self.get_node(node_index, copy=False)
        right = self.get_node(node_index + 1, copy=False)
        tn.check_connected((left, right))
        edge = tn.get_shared_edges(left, right).pop()
        return edge.dimension

    def bond_dimensions(self) -> List[int]:
        """Returns the bond dimension of each edge in the MPS."""
        return [self.bond_dimension_of(i) for i in range(self._nqudits - 1)]

    def max_bond_dimension_of(self, edge_index: int) -> int:
        """Returns the maximum bond dimension of the right edge
        of the node at the given index.

        Args:
            edge_index: Index of the node.
                The returned bond dimension is that of the right edge
                of the given node.
                Negative indices count backwards from the right of the MPS.
        """
        if edge_index >= self._nqudits:
            raise ValueError(
                f"Edge index should be less than {self._nqudits} "
                f"but is {edge_index}."
            )
        return self._max_bond_dimensions[edge_index]

    def max_bond_dimensions(self) -> List[int]:
        """Returns the maximum bond dimensions of the MPS."""
        return self._max_bond_dimensions

    def is_valid(self) -> bool:
        """Returns true if the MPS is valid, else False.

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
        """Returns the nodes of the MPS.

        Args:
            copy: If True, a copy of the nodes are returned, else the actual
                nodes are returned.
        """
        if not copy:
            return self._nodes
        nodes_dict, _ = tn.copy(self._nodes)
        return list(nodes_dict.values())

    def get_node(self, node_index: int, copy: bool = True) -> tn.Node:
        """Returns the ith node in the MPS counting from the left.

        Args:
            node_index: Index of node to get.
            copy: If true, a copy of the node is returned,
                else the actual node is returned.
        """
        return self.get_nodes(copy)[node_index]

    def get_free_edge_of(self, node_index: int, copy: bool = True) -> tn.Edge:
        """Returns the free (dangling) edge of a node with specified index.
        
        Args:
            node_index: Specifies the node.
            copy: If True, returns a copy of the edge.
                If False, returns the actual edge.
        """
        return self.get_node(node_index, copy).get_all_dangling().pop()

    def get_left_connected_edge_of(
            self, node_index: int
    ) -> Union[tn.Edge, None]:
        """Returns the left connected edge of the specified node, if it exists.

        Args:
            node_index: Index of node to get the left connected edge of.
        """
        if node_index == 0:
            return None

        return tn.get_shared_edges(
            self._nodes[node_index], self._nodes[node_index - 1]
        ).pop()

    def get_right_connected_edge_of(
            self, node_index: int
    ) -> Union[tn.Edge, None]:
        """Returns the left connected edge of the specified node, if it exists.

        Args:
            node_index: Index of node to get the right connected edge of.
        """
        if node_index == self._nqudits - 1:
            return None

        return tn.get_shared_edges(
            self._nodes[node_index], self._nodes[node_index + 1]
        ).pop()

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

    def dagger(self):
        """Takes the dagger (conjugate transpose) of the MPS."""
        for i in range(self._nqudits):
            self._nodes[i].set_tensor(np.conj(self._nodes[i].tensor))

    def inner_product(self, other: 'MPS') -> np.complex:
        """Returns the inner product between self and other computed by
        contraction. Mathematically, the inner product is <self|other>.

        Args:
            other: Other MPS to take inner product with. This MPS is the "ket"
                in the inner product <self|other>.

        Raises:
            ValueError: If:
                * Number of qudits don't match in both MPS.
                * Qudit dimensions don't match in both MPS.
                * Either MPS is invalid
        """
        if other._nqudits != self._nqudits:
            raise ValueError(
                f"Cannot compute inner product between self which has "
                f"{self._nqudits} qudits and other which has {other._nqudits} "
                f"qudits."
                "\nNumber of qudits must be equal."
            )

        if other._qudit_dimension != self._qudit_dimension:
            raise ValueError(
                f"Cannot compute inner product between self which has qudit"
                f"dimension {self._qudit_dimension} and other which as qudit"
                f"dimension {other._qudit_dimension}."
                "Qudit dimensions must be equal."
            )

        if not self.is_valid():
            raise ValueError("MPS is invalid.")

        if not other.is_valid():
            raise ValueError("Other MPS is invalid.")

        a = self.get_nodes(copy=True)
        b = other.get_nodes(copy=True)
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
        return np.complex(fin.tensor)

    def norm(self) -> float:
        """Returns the norm of the MPS computed by contraction."""
        return np.sqrt(self.inner_product(self).real)

    def renormalize(self, to_norm: float = 1.0) -> None:
        """Renormalizes the MPS.

        Args:
            to_norm: The new norm of the MPS.

        Raises:
            ValueError: If to_norm is negative or too close to zero, or if
                the current norm of the MPS is too close to zero.
        """
        if to_norm < 0.:
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
                (to_norm / norm)**(1 / self.nqudits) * node.tensor
            )

    def reduced_density_matrix(
        self,
        node_indices: Union[int, Sequence[int]]
    ) -> np.ndarray:
        """Computes the reduced density matrix of the MPS on the given nodes.

        Args:
            node_indices: Node index, or list of node indices, to keep.
                Indices not in node_indices are traced out.

        Raises:
            ValueError: If the node_indices contain duplicate indices.
            IndexError: If the indices are out of bounds for the MPS.
        """
        try:
            node_indices = iter(node_indices)
        except TypeError:
            node_indices = [node_indices]
        node_indices = tuple(node_indices)

        if len(set(node_indices)) < len(node_indices):
            raise ValueError("Node indices contains duplicates.")

        if min(node_indices) < 0 or max(node_indices) > self._nqudits - 1:
            raise IndexError("One or more invalid node indices.")

        ket = self.copy()
        bra = self.copy()
        bra.dagger()
        ket_edges = [node.get_all_dangling().pop() for node in ket._nodes]
        bra_edges = [node.get_all_dangling().pop() for node in bra._nodes]

        # Store ordered free edges to reorder edges in the final tensor
        ket_free_edges = []
        bra_free_edges = []
        for i in node_indices:
            ket_free_edges.append(ket_edges[i])
            bra_free_edges.append(bra_edges[i])

        for i in range(self._nqudits):
            # If this node is not in node_indices, trace it out
            if i not in node_indices:
                _ = tn.connect(ket_edges[i], bra_edges[i])

            # Contract while allowing outer product for unconnected nodes
            mid = tn.contract_between(
                ket._nodes[i], bra._nodes[i], allow_outer_product=True
            )

            if i < self._nqudits - 1:
                new = tn.contract_between(mid, ket._nodes[i + 1])
                ket._nodes[i + 1] = new

        mid.reorder_edges(ket_free_edges + bra_free_edges)
        n = len(node_indices)
        d = self._qudit_dimension
        return np.reshape(mid.tensor, newshape=(d**n, d**n))

    def _sample(self) -> Sequence[int]:
        """Returns a string of measured states at each site
        by sampling once from the MPS.
        """
        string = []
        states = list(range(self._qudit_dimension))
        copy = self.copy()
        for i in range(self._nqudits):
            qubit = self.reduced_density_matrix(i).diagonal().real
            string.append(np.random.choice(states, size=1, p=qubit)[0])
            state = computational_basis_state(
                string[-1], dim=self._qudit_dimension
            )
            # print("Qubit:", qubit)
            # print("Measured:", string[-1])
            edge = tn.connect(
                copy._nodes[i].get_all_dangling().pop(),
                state.get_all_dangling().pop()
            )
            mid = tn.contract(edge)
            if i < self._nqudits - 1:
                new = tn.contract_between(mid, copy._nodes[i + 1])
                copy._nodes[i + 1] = new
        # print("Final node:")
        # print(mid.tensor)
        assert len(mid.edges) == 0
        prob = abs(mid.tensor)**2
        print("String:", string)
        print("Prob:", np.round(prob, 2))
        return string

    def sample(self, nsamples: int) -> List[Sequence[int]]:
        """Samples from the MPS, returning a list of (bit)strings of measured
        states on each site.

        Args:
            nsamples: Number of times to sample from the MPS.

        Raises: ValueError: If nsamples is negative or non-integer.
        """
        if not isinstance(nsamples, int):
            raise ValueError(
                f"Arg nsamples should be an int but is a {type(nsamples)}."
            )

        if nsamples <= 0:
            raise ValueError(
                f"Arg nsamples should be positive but is {nsamples}."
            )
        return [self._sample() for _ in range(nsamples)]

    def expectation(self, observable: MPSOperation) -> float:
        """Returns the expectation value of an observable <mps|observable|mps>.

        Args:
            observable: Hermitian operator expressed as an MPSOperation.
                Example:
                    To compute the expectation of H \otimes I on a two-qubit MPS

                    >>> observable = MPSOperation(mpsim.hgate(), 0)
                    >>> mps.expectation(observable)

        Raises:
            ValueError: If the observable is not Hermitian.
        """
        if not observable.is_hermitian():
            raise ValueError("Observable is not Hermitian.")

        if observable.qudit_dimension != self._qudit_dimension:
            obs_dim = observable.qudit_dimension
            mps_dim = self._qudit_dimension
            raise ValueError(
                f"Dimension mismatch between observable and MPS. "
                f"Observable is ({obs_dim}, {obs_dim}) but MPS has qudit "
                f"dimension {mps_dim}."
            )

        mps_copy = self.copy()
        mps_copy.apply(observable)
        return self.inner_product(mps_copy).real

    def apply_one_qudit_gate(
        self,
        gate: tn.Node,
        node_index: int,
        **kwargs
    ) -> None:
        """Applies a single qubit gate to a specified node.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
            node_index: Index of tensor (qubit) in the MPS to apply
                the single qubit gate to.

        Keyword Args:
            ortho_after_non_unitary (bool): If True, orthonormalizes edge(s)
                of the node after applying a non-unitary gate.
            renormalize_after_non_unitary (bool): If True, renormalize the MPS
                after applying a non-unitary gate.

        Notes:
            Edge convention.
                Gate edge 1: Connects to MPS node.
                Gate edge 0: Becomes free edge of new MPS node.

        Raises:
            ValueError:
                On invalid MPS, invalid index, invalid gate, and edge dimension
                mismatch between gate edges and MPS qudit edges.
        """
        if not self.is_valid():
            raise ValueError("MPS is invalid.")

        if node_index not in range(self._nqudits):
            raise ValueError(
                f"Input tensor index={node_index} is out of bounds for"
                f" an MPS on {self._nqudits} qudits."
            )

        if (len(gate.get_all_dangling()) != 2
                or len(gate.get_all_nondangling()) != 0):
            raise ValueError(
                "Single qudit gate must have two free edges"
                " and zero connected edges."
            )

        if gate.get_edge(0).dimension != gate.get_edge(1).dimension:
            raise ValueError("Gate edge dimensions must be equal.")

        if gate.get_edge(0).dimension != self._qudit_dimension:
            raise ValueError(
                f"Gate edges have dimension {gate.get_edge(0).dimension} "
                f"but should have MPS qudit dimension = {self._qudit_dimension}"
            )

        # Parse the keyword arguments
        renormalize_after_non_unitary = True
        ortho_after_non_unitary = True
        if kwargs.get("renormalize_after_non_unitary") is False:
            renormalize_after_non_unitary = False
        if kwargs.get("ortho_after_non_unitary") is False:
            ortho_after_non_unitary = False

        # Store the norm for optional renormalization after non-unitary gate
        if not is_unitary(gate) and renormalize_after_non_unitary:
            norm = self.norm()

        # Connect the MPS and gate edges
        mps_edge = list(self._nodes[node_index].get_all_dangling())[0]
        gate_edge = gate[1]
        connected = tn.connect(mps_edge, gate_edge)

        # Contract the edge to get the new tensor
        new = tn.contract(connected, name=self._nodes[node_index].name)
        self._nodes[node_index] = new

        # Optional orthonormalization after a non-unitary gate
        if not is_unitary(gate) and ortho_after_non_unitary:
            # Edge case: Left-most node
            if node_index == 0:
                self.orthonormalize_right_edge_of(node_index)

            # Edge case: Right-most node
            elif node_index == self._nqudits - 1:
                self.orthonormalize_left_edge_of(node_index)

            # General case
            else:
                self.orthonormalize_right_edge_of(node_index)
                self.orthonormalize_left_edge_of(node_index)

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

    def apply_one_qudit_gate_to_all(self, gate: tn.Node) -> None:
        """Applies a single qudit gate to all tensors in the MPS.

        Args:
            gate: Single qudit gate to apply. A tensor with two free indices.
        """
        for i in range(self._nqudits):
            self.apply_one_qudit_gate(gate, i)

    def apply_two_qudit_gate(
        self, gate: tn.Node, node_index1: int, node_index2: int, **kwargs
    ) -> None:
        """Applies a two qubit gate to the specified nodes.

        Args:
            gate: Two qubit gate to apply. See Notes for the edge convention.
            node_index1: Index of first node in the MPS the gate acts on.
            node_index2: Index of second node in the MPS the gate acs on.

        Keyword Arguments:
            keep_left_canonical: After performing an SVD on the new node to
                obtain U, S, and Vdag, S is grouped with Vdag to form the
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

            ensures the edge convention below is satisfied.

            Gate edge convention (assuming indexA < indexB)
                gate edge 2: Connects to tensor at indexA.
                gate edge 3: Connects to tensor at indexB.
                gate edge 0: Becomes free index of new tensor at indexA.
                gate edge 1: Becomes free index of new tensor at indexB.

            If indexA > indexB, 0 <--> 1 and 2 <--> 3.

        Raises:
            ValueError: On the following:
                * Invalid MPS.
                * Invalid indices (equal or out of bounds).
                * Invalid two-qudit gate.
        """
        if not self.is_valid():
            raise ValueError("MPS is not valid.")

        if (node_index1 not in range(self._nqudits)
                or node_index2 not in range(self.nqudits)):
            raise ValueError(
                f"Input tensor indices={(node_index1, node_index2)} are out of "
                f"bounds for an MPS on {self._nqudits} qudits."
            )

        if node_index1 == node_index2:
            raise ValueError("Node indices cannot be identical.")

        if (len(gate.get_all_dangling()) != 4
                or len(gate.get_all_nondangling()) != 0):
            raise ValueError(
                "Two qubit gate must have four free edges"
                " and zero connected edges."
            )

        edge_dimensions = set([edge.dimension for edge in gate.edges])
        if len(edge_dimensions) != 1:
            raise ValueError("All gate edges must have the same dimension.")

        if edge_dimensions.pop() != self._qudit_dimension:
            raise ValueError(
                f"Gate edges have dimension {gate.get_edge(0).dimension} "
                f"but should have MPS qudit dimension = {self._qudit_dimension}"
            )

        # Flip the "control"/"target" gate edges and tensor edges if needed
        if node_index2 < node_index1:
            gate.reorder_edges([gate[1], gate[0], gate[3], gate[2]])
            node_index1, node_index2 = node_index2, node_index1

        # Swap tensors until adjacent if necessary
        invert_swap_network = False
        if node_index1 < node_index2 - 1:
            invert_swap_network = True
            original_index1 = node_index1
            self.move_node_from_left_to_right(
                node_index1, node_index2 - 1, **kwargs
            )
            node_index1 = node_index2 - 1

        # Connect the MPS tensors to the gate edges
        _ = tn.connect(
            self.get_free_edge_of(node_index=node_index1, copy=False),
            gate.get_edge(2)
        )
        _ = tn.connect(
            self.get_free_edge_of(node_index=node_index2, copy=False),
            gate.get_edge(3)
        )

        # Store the free edges of the gate
        left_gate_edge = gate.get_edge(0)
        right_gate_edge = gate.get_edge(1)

        # Contract the tensors in the MPS
        new_node = tn.contract_between(
            self._nodes[node_index1], self._nodes[node_index2]
        )

        # Flatten the two edges from the MPS node to the gate node
        node_gate_edge = tn.flatten_edges_between(new_node, gate)

        # Contract the flattened edge to get a new single MPS node
        new_node = tn.contract(node_gate_edge)

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
            if index <= node_index1:
                left_connected_edge = connected_edge
            else:
                right_connected_edge = connected_edge

        # ================================================
        # Do the SVD to split the single MPS node into two
        # ================================================
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
                round(fraction * self.max_bond_dimension_of(
                    min(node_index1, node_index2)
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
        new_left.name = self._nodes[node_index1].name
        new_right.name = self._nodes[node_index2].name

        self._nodes[node_index1] = new_left
        self._nodes[node_index2] = new_right

        # Invert the Swap network, if necessary
        if invert_swap_network:
            self.move_node_from_right_to_left(
                node_index1, original_index1, **kwargs
            )

        # TODO: Remove. This is only for convenience in benchmarking.
        self._norms.append(self.norm())

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
            # TODO: SWAP is only for qubits
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
            # TODO: SWAP is only for qubits
            self.swap(current_node_index - 1, current_node_index, **kwargs)
            current_node_index -= 1

    def apply(
        self,
        operations: Union[MPSOperation, Sequence[MPSOperation]],
        **kwargs,
    ) -> None:
        """Apply operations to the MPS.

        Args:
            operations: (Sequence of) valid MPS Operation(s) to apply.

        Keyword Args:
            For one qudit gates:


            For two qudit gates:

        Raises:
            ValueError: On an invalid operation.
        """
        try:
            operations = iter(operations)
        except TypeError:
            operations = (operations,)

        # TODO: Parallelize application of operations
        for op in operations:
            self._apply_mps_operation(op, **kwargs)

    def _apply_mps_operation(self, operation: MPSOperation, **kwargs) -> None:
        """Applies the MPS Operation to the MPS.

        Args:
            operation: Valid MPS Operation to apply to the MPS.
        """
        if not isinstance(operation, MPSOperation):
            raise TypeError(
                "Argument operation should be of type MPSOperation but is "
                f"of type {type(operation)}."
            )
        if not operation.is_valid():
            raise ValueError("Input MPS Operation is not valid.")

        if operation.is_single_qudit_operation():
            self.apply_one_qudit_gate(
                operation.node(), *operation.qudit_indices, **kwargs
            )
        elif operation.is_two_qudit_operation():
            self.apply_two_qudit_gate(
                operation.node(), *operation.qudit_indices, **kwargs
            )
        else:
            raise ValueError(
                "Only one-qudit and two-qudit gates are supported. "
                "To apply a gate on three or more qudits, the gate must be "
                "compiled into a sequence of one- and two-qudit gates."
            )

    # TODO: Remove single qubit gates -- these don't generalize to qudits.
    def x(self, index: int) -> None:
        """Applies a NOT (Pauli-X) gate to a qubit specified by the index.

        If index == -1, the gate is applied to all qubits.

        Args:
            index: Index of qubit (tensor) to apply X gate to.
        """
        if index == -1:
            self.apply_one_qudit_gate_to_all(xgate())
        else:
            self.apply_one_qudit_gate(xgate(), index)

    def h(self, index: int) -> None:
        """Applies a Hadamard gate to a qubit specified by the index.

        If index == -1, the gate is applied to all qubits.

        Args:
            index: Index of qubit (tensor) to apply X gate to.
        """
        if index == -1:
            self.apply_one_qudit_gate_to_all(hgate())
        else:
            self.apply_one_qudit_gate(hgate(), index)

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
                self.apply_one_qudit_gate(
                    rgate(seed, angle_scale), i,
                )
        else:
            self.apply_one_qudit_gate(rgate(seed, angle_scale), index)

    # TODO: Remove. This doesn't generalize to qudits.
    def cnot(self, a: int, b: int, **kwargs) -> None:
        """Applies a CNOT gate with qubit indexed `a` as control
        and qubit indexed `b` as target.
        """
        self.apply_two_qudit_gate(cnot(), a, b, **kwargs)

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
        self.apply_two_qudit_gate(swap(), a, b, **kwargs)

    def copy(self) -> 'MPS':
        """Returns a copy of the MPS."""
        return self.__copy__()

    def __str__(self):
        return "----".join(str(tensor) for tensor in self._nodes)

    def __eq__(self, other: 'MPS'):
        if not isinstance(other, MPS):
            return False
        if self is other:
            return True
        if not self.is_valid():
            raise ValueError(
                "MPS is invalid and cannot be compared to another MPS."
            )
        if not other.is_valid():
            raise ValueError(
                "Other MPS is invalid."
            )
        if (other._qudit_dimension != self._qudit_dimension or
                other._nqudits != self._nqudits):
            return False
        for i in range(self._nqudits):
            if not np.allclose(
                    self.get_node(i).tensor, other.get_node(i).tensor
            ):
                return False
            if i > 0:
                if (self.get_left_connected_edge_of(i).dimension !=
                        other.get_left_connected_edge_of(i).dimension):
                    return False
            if i < self._nqudits - 1:
                if (self.get_right_connected_edge_of(i).dimension !=
                        other.get_right_connected_edge_of(i).dimension):
                    return False
        return True

    def __copy__(self):
        new = MPS(self._nqudits, self._qudit_dimension, self._prefix)
        new._nodes = self.get_nodes(copy=True)
        return new
