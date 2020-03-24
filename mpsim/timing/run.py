"""Implements the algorithm described in [1] and times the performance.

References:
    [1] Zhou, Stoudenmire, and Waintal, "What limits the simulation of quantum computers?", arXiv:2002.07730, 2020.
"""

import sys

from mpsim import simulate


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nqubits = int(sys.argv[1])
        print(f"\nUsing {nqubits} qubits.")
    else:
        nqubits = 10
        print(f"\nUsing default nqubits = {nqubits}.")

    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
        print("Using depth =", depth)
    else:
        depth = 20
        print("Using default depth =", depth)

    if len(sys.argv) > 3:
        keep = int(sys.argv[3])
        print(f"Keeping {keep} singular values for every two-qubit gate.")
    else:
        keep = None
        print(f"Keeping all singular values for every two-qubit gate.")

    print("\nSimulating algorithm...")
    mps = simulate(nqubits, depth, keep, verbose=True)
