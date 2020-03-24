"""Implements the algorithm described in [1] and performs various tests.

References:
    [1] Zhou, Stoudenmire, and Waintal, "What limits the simulation of quantum computers?", arXiv:2002.07730, 2020.
"""

import sys
import time
from typing import Union

from mps import MPS


def simulate(nqubits: int, depth: int, keep: Union[None, int], verbose: bool = False) -> MPS:
    """Simulates the algorithm for a given number of qubits and depth."""
    mps = MPS(nqubits)
    start = time.time()
    for d in range(depth):
        if verbose:
            print("At depth =", d + 1)
        mps.r(-1)
        mps.sweep_cnots_left_to_right(keep=keep)
        mps.r(-1)
        mps.sweep_cnots_right_to_left(keep=keep)
    runtime_sec = time.time() - start
    print("\nCompleted in", round(runtime_sec, 3), "seconds.")
    return mps


if __name__ == "__main__":
    print("=" * 40)
    print("Simulating Waintal circuit")
    print("=" * 40)

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
