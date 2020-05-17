# MPSim

Package for using Matrix Product States (MPS) to simulate quantum circuits.

# Installation

After cloning the repository, run

```bash
pip install -e .
```

in the directory with `setup.py`.

# Getting started

The main object is `mpsim.MPS` which defines a matrix product state of qudits initialized to the all zero state.

An `MPS` can be acted on by aribtrary one-qudit and two-qudit operations. Operations on >=3 qubits are not supported and
must be compiled into a sequence of one- and two-qudit operations.

An `MPSOperation` consists of a gate and tuple of indices specifying which tensor(s) the gate acts on in the MPS.
Gates must be expressed as `TensorNetwork.Node` objects with the appropriate number of edges.
Some common gates are defined in `mpsim.gates`.

The following program initializes an MPS in the |00> state and prepares a Bell state.

```python
import mpsim

mps = mpsim.MPS(nqudits=2)
mps.h(0)
mps.cnot(0, 1)

print(mps.wavefunction)
# Displays [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

# Two-qubit gate options

The number of singular values kept for a two-qubit gate can be set by the keyword argument `maxsvals`.

The following program prepares the same Bell state but only keeps one singular value after the CNOT.

```python
import mpsim

mps = mpsim.MPS(nqudits=2)
mps.h(0)
mps.cnot(0, 1, maxsvals=1)

print(mps.wavefunction)
# Displays [0.70710674+0.j 0.        +0.j 0.        +0.j 0.        +0.j]
```

Note that the wavefunction after truncation is not normalized. An `MPS` can be renormalized at any time by calling the
`MPS.renormalize()` method.

# Cirq integration

Circuits defined in [Cirq](https://github.com/quantumlib/Cirq) can be simulated with MPS as follows.

```python
import cirq
from mpsim.mpsim_cirq.simulator import MPSimulator

# Define the circuit
qreg = cirq.LineQubit.range(2)
circ = cirq.Circuit(
    cirq.ops.H.on(qreg[0]),
    cirq.ops.CNOT(*qreg)
)

# Do the simulation using the MPS Simulator
sim = MPSimulator()
mps = sim.simulate(circ)
print(mps.wavefunction)
# Displays [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

One can truncate singular values for two-qubit operations by passing in options to the `MPSimulator`.

```python
sim = MPSimulator(options={"maxsvals": 1})
mps = sim.simulate(circ)
print(mps.wavefunction)
# Displays [0.70710674+0.j 0.        +0.j 0.        +0.j 0.        +0.j]
```

See `help(MPSimulator)` for a full list of options.


