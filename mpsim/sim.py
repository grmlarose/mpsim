"""Defines functions for simulating circuits using MPS."""

import time
from typing import Optional, Union

import numpy as np

from mpsim import MPS


def simulate(
    nqubits: int,
    depth: int,
    verbose: bool = False,
    seed: Optional[int] = None,
    small_angles: bool = False,
    **kwargs,
) -> MPS:
    """Simulates a Waintall circuit using MPS for a given number of qubits and depth.

    Args:
        nqubits: Number of qubits in the circuit.
        depth: Depth of the circuit. See [1] for details.
        seed: Seed for random number generator used in random single qubit rotations.
        small_angles: Option to make single qubit rotations close to identity.


    Keyword Args:
        fraction (float): Number of singular values to keep expressed as a fraction of the maximum bond dimension.
        maxsvals (int): Number of singular values to keep for every two-qubit gate.
    """
    mps = MPS(nqubits)

    if seed:
        np.random.seed(seed)

    if verbose:
        print("=" * 40)
        print("Simulating Waintal circuit on {nqubits} qubits")
        print("=" * 40)

    start = time.time()
    for d in range(depth):
        if verbose:
            print(f"At depth {d + 1} / {depth}")
        mps.r(-1, small_angles=small_angles)
        mps.sweep_cnots_left_to_right(**kwargs)
        mps.r(-1, small_angles=small_angles)
        mps.sweep_cnots_right_to_left(**kwargs)
    runtime_sec = time.time() - start

    if verbose:
        print("\nCompleted in", round(runtime_sec, 3), "seconds.")
    return mps
