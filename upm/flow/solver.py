"""
solver.py
=========
Solves the UPM pipe network flow equation:
    [K] {Phi} = {Qs}

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Ren et al. (2017) - Unified Pipe Network Method
"""

import numpy as np
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import issparse


def solve_pressure(K, Qs, method='direct', tolerance=1e-10):
    """
    Solve [K]{Phi} = {Qs} for nodal pressures.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Global conductance matrix with BCs applied.
    Qs : numpy.ndarray
        Source vector with BCs applied.
    method : str
        'direct' or 'iterative'
    tolerance : float
        Convergence tolerance for iterative solver.

    Returns
    -------
    numpy.ndarray
        Nodal pressure vector (Pa).
    """
    if not issparse(K):
        raise ValueError("K must be a sparse matrix.")

    n_nodes = K.shape[0]
    print(f"  Solving {n_nodes}x{n_nodes} system "
          f"using {method} solver...")

    if method == 'direct':
        Phi = spsolve(K, Qs)

    elif method == 'iterative':
        Phi, info = cg(K, Qs, tol=tolerance, maxiter=10*n_nodes)
        if info != 0:
            raise ValueError(
                f"Iterative solver did not converge. "
                f"Info code: {info}"
            )
    else:
        raise ValueError(
            f"Unknown solver method: {method}. "
            f"Use 'direct' or 'iterative'."
        )

    if np.any(np.isnan(Phi)) or np.any(np.isinf(Phi)):
        raise ValueError(
            "Solver returned NaN or Inf values.\n"
            "Check boundary conditions and matrix assembly."
        )

    print(f"  Solved successfully!")
    print(f"  Pressure range: "
          f"{Phi.min():.3e} to {Phi.max():.3e} Pa")

    return Phi


def compute_pipe_flow_rates(pipes, nodes, Phi):
    """
    Compute flow rate in each pipe.

    Q(i,j) = K(i,j) * (Phi_i - Phi_j)

    Parameters
    ----------
    pipes : list of Pipe
    nodes : list of Node
    Phi : numpy.ndarray
        Nodal pressure solution.

    Returns
    -------
    numpy.ndarray
        Flow rates (m3/s).
    """
    Q = np.zeros(len(pipes))
    for idx, pipe in enumerate(pipes):
        i = pipe.node_i
        j = pipe.node_j
        Q[idx] = pipe.conductance * (Phi[i] - Phi[j])
    return Q


def assign_pressures_to_nodes(nodes, Phi):
    """
    Assign solved pressures back to node objects.

    Parameters
    ----------
    nodes : list of Node
    Phi : numpy.ndarray

    Returns
    -------
    list of Node
    """
    for i, node in enumerate(nodes):
        node.pressure = float(Phi[i])
    return nodes


def compute_total_flow(pipes, Q, inlet_nodes):
    """
    Compute total inflow at inlet boundary nodes.

    Parameters
    ----------
    pipes : list of Pipe
    Q : numpy.ndarray
    inlet_nodes : list of int

    Returns
    -------
    float
        Total volumetric flow rate (m3/s).
    """
    total_Q = 0.0
    for idx, pipe in enumerate(pipes):
        if pipe.node_i in inlet_nodes:
            total_Q += Q[idx]
        elif pipe.node_j in inlet_nodes:
            total_Q -= Q[idx]
    return total_Q


def print_solution_summary(nodes, pipes, Phi, Q):
    """
    Print a readable summary of the flow solution.

    Parameters
    ----------
    nodes : list of Node
    pipes : list of Pipe
    Phi : numpy.ndarray
    Q : numpy.ndarray
    """
    print("=" * 50)
    print("  FLOW SOLUTION SUMMARY")
    print("=" * 50)

    print(f"\n  PRESSURE FIELD:")
    print(f"    Min  : {Phi.min():.3e} Pa")
    print(f"    Max  : {Phi.max():.3e} Pa")
    print(f"    Mean : {Phi.mean():.3e} Pa")

    print(f"\n  FLOW RATES:")
    print(f"    Min |Q|  : {np.abs(Q).min():.3e} m3/s")
    print(f"    Max |Q|  : {np.abs(Q).max():.3e} m3/s")
    print(f"    Mean |Q| : {np.abs(Q).mean():.3e} m3/s")

    print(f"\n  NODE PRESSURES:")
    for node in nodes:
        bc_flag = " <- BC" if node.is_boundary else ""
        print(f"    Node {node.node_id:2d} "
              f"({node.node_type:20}): "
              f"P = {node.pressure:.3e} Pa{bc_flag}")

    print(f"\n  PIPE FLOW RATES (top 5):")
    sorted_idx = np.argsort(np.abs(Q))[::-1]
    for idx in sorted_idx[:5]:
        pipe = pipes[idx]
        print(f"    Pipe {pipe.pipe_id:2d} "
              f"(node {pipe.node_i}->{pipe.node_j}): "
              f"Q = {Q[idx]:.3e} m3/s")

    print("=" * 50)