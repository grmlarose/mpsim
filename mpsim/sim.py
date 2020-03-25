"""Defines functions for simulating circuits using MPS."""

import time
from typing import Union

from mpsim import MPS


def simulate(
    nqubits: int,
    depth: int,
    fraction: float,
    verbose: bool = False,
) -> MPS:
    """Simulates a Waintall circuit using MPS for a given number of qubits and depth.

    Args:
        nqubits: Number of qubits in the circuit.
        depth: Depth of the circuit. See [1] for details.
        fraction: Number of singular values to keep expressed as a fraction of the maximum bond dimension.

    """
    mps = MPS(nqubits)
    if verbose:
        print("=" * 40)
        print("Simulating Waintall circuit")
        print("=" * 40)
    start = time.time()
    for d in range(depth):
        if verbose:
            print(f"At depth {d + 1} / {depth}")
        mps.r(-1)
        mps.sweep_cnots_left_to_right(fraction=fraction)
        mps.r(-1)
        mps.sweep_cnots_right_to_left(fraction=fraction)
    runtime_sec = time.time() - start
    if verbose:
        print("\nCompleted in", round(runtime_sec, 3), "seconds.")
    return mps
