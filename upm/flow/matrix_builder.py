"""
matrix_builder.py
=================
Assembles the global conductance matrix [K] for the UPM pipe network.

The governing equation for steady saturated flow:
    [K] {Phi} = {Qs}

where:
    [K]   = global conductance matrix (n_nodes x n_nodes)
    {Phi} = nodal flow potential vector (unknown)
    {Qs}  = nodal source/sink vector

Assembly rules (from Ren et al. 2017):
    Diagonal entry    K[i,i] = sum of all pipe conductances at node i
    Off-diagonal entry K[i,j] = -K(i,j) for pipe connecting i and j

The matrix is:
    Symmetric        ✓
    Positive-definite ✓ (after applying boundary conditions)
    Sparse           ✓ (most entries are zero)

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Ren et al. (2017) - Unified Pipe Network Method
    Equation (4): [K]{Phi} = {Qs}
    Equation (5): K_ii = sum K(i,j)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def build_K_matrix(nodes, pipes):
    """
    Assemble global conductance matrix [K].

    Assembly rules from Ren et al. (2017):
        K[i,i] += K_pipe    for each pipe at node i
        K[i,j] -= K_pipe    for pipe connecting nodes i and j
        K[j,i] -= K_pipe    (symmetric)
        K[j,j] += K_pipe    for each pipe at node j

    Parameters
    ----------
    nodes : list of Node
        All nodes in the pipe network.
    pipes : list of Pipe
        All pipes with conductance assigned.

    Returns
    -------
    scipy.sparse.csr_matrix
        Global conductance matrix [K], shape (n_nodes, n_nodes).

    Raises
    ------
    ValueError
        If pipes have no conductance assigned.

    Example
    -------
    K = build_K_matrix(nodes, pipes)
    print(f"Matrix shape: {K.shape}")
    print(f"Non-zero entries: {K.nnz}")
    """
    n_nodes = len(nodes)

    if n_nodes == 0:
        raise ValueError("No nodes in network.")
    if not pipes:
        raise ValueError("No pipes in network.")

    # check conductances assigned
    if all(p.conductance == 0.0 for p in pipes):
        raise ValueError(
            "All pipe conductances are zero. "
            "Run assign_conductances() first."
        )

    # use lil_matrix for efficient assembly
    K = lil_matrix((n_nodes, n_nodes), dtype=float)

    for pipe in pipes:
        i = pipe.node_i
        j = pipe.node_j
        k = pipe.conductance

        # diagonal entries
        K[i, i] += k
        K[j, j] += k

        # off-diagonal entries
        K[i, j] -= k
        K[j, i] -= k

    # convert to CSR for efficient solving
    K_csr = csr_matrix(K)

    print(f"  Assembled [K] matrix: {n_nodes}x{n_nodes}")
    print(f"  Non-zero entries    : {K_csr.nnz}")
    print(f"  Sparsity            : "
          f"{100*(1 - K_csr.nnz/n_nodes**2):.1f}%")

    return K_csr


def build_Qs_vector(nodes):
    """
    Build source/sink vector {Qs}.

    For most nodes Qs = 0 (no source or sink).
    Sources and sinks are applied via boundary conditions.

    Parameters
    ----------
    nodes : list of Node
        All nodes in the pipe network.

    Returns
    -------
    numpy.ndarray
        Source vector {Qs}, shape (n_nodes,).

    Example
    -------
    Qs = build_Qs_vector(nodes)
    print(f"Source vector shape: {Qs.shape}")
    """
    n_nodes = len(nodes)
    Qs = np.zeros(n_nodes, dtype=float)
    return Qs


def check_matrix_properties(K):
    """
    Check mathematical properties of the K matrix.
    Useful for debugging before solving.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Global conductance matrix.

    Returns
    -------
    dict
        Dictionary of matrix properties.

    Example
    -------
    props = check_matrix_properties(K)
    print(props)
    """
    K_dense = K.toarray()
    n       = K.shape[0]

    # check symmetry
    is_symmetric = np.allclose(K_dense, K_dense.T, atol=1e-10)

    # check diagonal positive
    diag = np.diag(K_dense)
    diag_positive = np.all(diag >= 0)

    # check row sums (should be zero for interior nodes)
    row_sums = np.abs(K_dense.sum(axis=1))
    max_row_sum = row_sums.max()

    # check for isolated nodes (zero diagonal)
    isolated = np.sum(diag == 0)

    props = {
        'shape'         : K.shape,
        'nnz'           : K.nnz,
        'is_symmetric'  : is_symmetric,
        'diag_positive' : diag_positive,
        'max_row_sum'   : max_row_sum,
        'isolated_nodes': isolated,
        'min_diagonal'  : diag.min(),
        'max_diagonal'  : diag.max()
    }

    return props


def print_matrix_summary(K, nodes=None):
    """
    Print a readable summary of the K matrix.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Global conductance matrix.
    nodes : list of Node, optional
        Node list for additional info.
    """
    props = check_matrix_properties(K)

    print("=" * 50)
    print("  [K] MATRIX SUMMARY")
    print("=" * 50)
    print(f"  Shape           : {props['shape']}")
    print(f"  Non-zeros       : {props['nnz']}")
    print(f"  Sparsity        : "
          f"{100*(1-props['nnz']/K.shape[0]**2):.1f}%")
    print(f"  Symmetric       : {props['is_symmetric']}")
    print(f"  Diag positive   : {props['diag_positive']}")
    print(f"  Isolated nodes  : {props['isolated_nodes']}")
    print(f"  Max row sum     : {props['max_row_sum']:.2e}")
    print(f"  Diag range      : "
          f"{props['min_diagonal']:.2e} to "
          f"{props['max_diagonal']:.2e}")

    if props['isolated_nodes'] > 0:
        print(f"\n  WARNING: {props['isolated_nodes']} isolated nodes!")
        print(f"  These nodes have no pipe connections.")
        print(f"  Check your fracture network connectivity.")

    if not props['is_symmetric']:
        print(f"\n  WARNING: Matrix is not symmetric!")
        print(f"  Check pipe assembly.")

    print("=" * 50)