"""Defines matrix product state class."""

from typing import List, Optional, Sequence, Union

import numpy as np
import tensornetwork as tn

from mpsim.gates import hgate, rgate, xgate, cnot, swap, is_unitary
from mpsim.mpsim_cirq.circuits import MPSOperation


class MPS:
    """Matrix Product State (MPS) for simulating (noisy) quantum circuits."""

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

    def apply_one_qubit_gate(self, gate: tn.Node, index: int) -> None:
        """Applies a single qubit gate to a specified node.

        Args:
            gate: Single qubit gate to apply. A tensor with two free indices.
            index: Index of tensor (qubit) in the MPS to apply
                    the single qubit gate to.
        """
        if not self.is_valid():
            raise ValueError("Input mpslist does not define a valid MPS.")

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

        # Connect the MPS and gate edges
        mps_edge = list(self._nodes[index].get_all_dangling())[0]
        gate_edge = gate[1]  # TODO: Is this the correct edge to use (always)?
        connected = tn.connect(mps_edge, gate_edge)

        # Contract the edge to get the new tensor
        new = tn.contract(connected, name=self._nodes[index].name)
        self._nodes[index] = new

        if not is_unitary(gate):
            # TODO: Is this the correct procedure for the right edge case?
            if index == self._nqudits - 1:
                index -= 1
            self._orthonormalize_edge(index)

    def _orthonormalize_edge(
            self, edge_index: int, maxsvals: int = 1
    ) -> None:
        """Performs SVD on a single node N to get N = U S V^dag. Sets the new
        node to be U and multiplies the node to the right by S V^dag.

        Args:
            edge_index: Index of edge to orthonormalize.
            maxsvals: Number of singular values to keep. This will be the
                      dimension of the new edge.
        """
        if not 0 <= edge_index < self._nqudits - 1:
            raise ValueError("Invalid edge index.")

        print("\n\nOriginal MPS nodes:")
        for node in self._nodes:
            print(node)
            print(node.tensor)
            print(*list(node.edges), sep="\n")
            print("\n\n")

        # Get the left node of the edge
        node = self.get_node(edge_index, copy=False)
        print("Node to SVD:")
        print(node)
        print(node.tensor)
        print(node.edges)

        # Get the left and right edges to do the SVD
        left_edges = [self.get_free_edge_of(edge_index, copy=False)]
        if self.get_left_connected_edge_of(edge_index):
            left_edges.append(self.get_left_connected_edge_of(edge_index))
        right_edges = [self.get_right_connected_edge_of(edge_index)]

        print("\nMy left edges are")
        print(left_edges)

        print("\nMy right edges are")
        print(right_edges)

        u, s, vdag, _ = tn.split_node_full_svd(
            node,
            left_edges=left_edges,
            right_edges=right_edges,
            max_singular_values=maxsvals
        )

        print("\n\nMy U node:")
        u.set_name("U")
        print(u)
        print(u.tensor)
        print(u.edges)

        print("\n\nMy S node:")
        s.set_name("S")
        print(s)
        print(s.tensor)
        print(s.edges)

        print("\n\nMy Vdag node:")
        vdag.set_name("Vdag")
        print(vdag)
        print(vdag.tensor)
        print(vdag.edges)

        # Set the new left node
        print(node.name)
        u.set_name(node.name)
        self._nodes[edge_index] = u
        self._nodes[edge_index].name = self._prefix + str(edge_index)

        # Mutlipy S and Vdag to the right
        temp = tn.contract_between(s, vdag)
        new_right = tn.contract_between(temp, self._nodes[edge_index + 1])
        new_right.name = self._nodes[edge_index + 1].name
        self._nodes[edge_index + 1] = new_right

        print("\n\nFinal MPS nodes:")
        for node in self._nodes:
            print(node)
            print(node.tensor)
            print(*list(node.edges), sep="\n")
            print()

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
            Gate edge convention:
                gate edge 0: Connects to tensor at indexA.
                gate edge 1: Connects to tensor at indexB.
                gate edge 2: Becomes free index of new tensor at indexA
                              after contracting.
                gate edge 3: Becomes free index of new tensor at indexB
                              after contracting.
        """
        if not self.is_valid():
            raise ValueError("Input mpslist does not define a valid MPS.")

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

        # Swap tensors until adjacent if necessary
        invert_swap_network = False
        if abs(indexA - indexB) != 1:
            invert_swap_network = True
            original_indexA = indexA
            self.move_node_from_left_to_right(indexA, indexB - 1)
            indexA = indexB - 1

        # Connect the MPS tensors to the gate edges
        if indexA < indexB:
            left_index = indexA
            right_index = indexB
        else:
            raise ValueError(f"IndexA must be less than IndexB.")

        _ = tn.connect(
            list(self._nodes[indexA].get_all_dangling())[0], gate.get_edge(0)
        )
        _ = tn.connect(
            list(self._nodes[indexB].get_all_dangling())[0], gate.get_edge(1)
        )

        # Store the free edges of the gate, using the docstring edge convention
        left_gate_edge = gate.get_edge(2)
        right_gate_edge = gate.get_edge(3)

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
            self.move_node_from_right_to_left(indexA, original_indexA)

        # TODO: Remove. This is only for convenience in benchmarking.
        self._fidelities.append(self.norm())

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
                mps_operation.node, *mps_operation.qudit_indices
            )
        elif mps_operation.is_two_qudit_operation():
            self.apply_two_qubit_gate(
                mps_operation.node, *mps_operation.qudit_indices, **kwargs
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
