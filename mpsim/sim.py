"""Defines functions for simulating circuits using MPS."""

import time
from typing import Union

from mpsim import MPS


def simulate(nqubits: int, depth: int, keep: Union[None, int], verbose: bool = False) -> MPS:
    """Simulates the algorithm for a given number of qubits and depth."""
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
        mps.sweep_cnots_left_to_right(keep=keep)
        mps.r(-1)
        mps.sweep_cnots_right_to_left(keep=keep)
    runtime_sec = time.time() - start
    if verbose:
        print("\nCompleted in", round(runtime_sec, 3), "seconds.")
    return mps
