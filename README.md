# MPSIM

Code for using Matrix Product States to SIMulate (noisy) quantum circuits.

# Contents

```
mpsim/
    | core
    | gates
    | sim
	mpsim_cirq/
		| circuits
		| simulator
```

# Installation

After cloning the repository, run

```bash
pip install -e .[development]
```

in the directory with `setup.py`.

# Getting started

The main object is `mpsim.MPS` which defines a one-dimensional matrix product state of qubits initialized to the all zero state.

An `MPS` can apply aribtrary one-qubit and nearest-neighbor two-qubit gates via the methods `MPS.apply_one_qubit_gate` and `MPS.apply_two_qubit_gate`. Gates must be expressed as `TensorNetwork.Node` objects with the appropriate number of edges. Some common gates are defined in `mpsim.gates` and convenience methods are included in `MPS` for common gates.

The following program initializes an MPS in the |00> state and prepares a Bell state.

```python
import mpsim

mps = mpsim.MPS(nqubits=2)
mps.h(0)
mps.cnot(0, 1)

print(mps.wavefunction)
# Displays [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

# Noisy simulation

By truncating the number of singular values kept after a two-qubit gate, (some model of) noisy simulation can be emulated.

The following program prepares the same Bell state but only keeps one singular value after the CNOT.

```python
import mpsim

mps = mpsim.MPS(nqubits=2)
mps.h(0)
mps.cnot(0, 1, maxsvals=1)

print(mps.wavefunction)
# Displays [0.70710674+0.j 0.        +0.j 0.        +0.j 0.        +0.j]
```

Note that the wavefunction after truncation is not normalized.

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

Truncation can be done by passing in options to the `MPSimulator`.

```python
sim = MPSimulator(options={"maxsvals": 1})
mps = sim.simulate(circ)
print(mps.wavefunction)
# Displays [0.70710674+0.j 0.        +0.j 0.        +0.j 0.        +0.j]
```

See `help(MPSimulator)` for a full list of options.


