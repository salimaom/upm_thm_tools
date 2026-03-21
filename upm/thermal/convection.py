"""
convection.py
=============
Heat convection assembly for UPM pipe network.

Heat convection by fluid flow in fractures:
    Q_conv = rho_w * cp_w * q * dT

where q is the Darcy flux (m3/s) in each pipe.

Discretized using upwind scheme:
    If Q > 0 (flow from i to j):
        heat flux = rho_w * cp_w * Q * T_i
    If Q < 0 (flow from j to i):
        heat flux = rho_w * cp_w * Q * T_j

This adds an asymmetric term to the thermal matrix:
    Kc[i,i] += rho_w * cp_w * max(Q, 0)
    Kc[i,j] -= rho_w * cp_w * max(Q, 0)
    Kc[j,j] += rho_w * cp_w * max(-Q, 0)
    Kc[j,i] -= rho_w * cp_w * max(-Q, 0)

The upwind scheme is stable for advection-dominated
problems (high Peclet number).

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Grenier et al. (2018) - Eq. 2 advection term
    Ren et al. (2017) - UPM convection
    Chen et al. (2018) - T-H coupling fractures
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def compute_peclet_number(rho_w, cp_w, Q, lambda_eff,
                           area, length):
    """
    Compute Peclet number for a pipe.

    Pe = rho_w * cp_w * |Q| * L / (lambda_eff * A)

    Pe >> 1 : advection dominated
    Pe << 1 : conduction dominated
    Pe ~ 1  : mixed regime

    Parameters
    ----------
    rho_w : float
        Water density (kg/m3).
    cp_w : float
        Water specific heat (J/kg/K).
    Q : float
        Flow rate in pipe (m3/s).
    lambda_eff : float
        Effective thermal conductivity (W/m/K).
    area : float
        Cross-sectional area (m2).
    length : float
        Pipe length (m).

    Returns
    -------
    float
        Peclet number (dimensionless).

    Example
    -------
    Pe = compute_peclet_number(1000, 4182, 1e-5, 0.6, 1e-6, 5.0)
    print(f"Pe = {Pe:.1f}")
    """
    if lambda_eff * area <= 0:
        return np.inf
    return (rho_w * cp_w * abs(Q) * length) / (lambda_eff * area)


def build_Kc_matrix(nodes, pipes, Q, config):
    """
    Assemble convective heat transport matrix [Kc].

    Uses upwind scheme for stability.

    For each pipe with flow rate Q[idx]:
        If Q > 0 (flow from node_i to node_j):
            Kc[i,i] += rho_w * cp_w * Q
            Kc[j,i] -= rho_w * cp_w * Q
        If Q < 0 (flow from node_j to node_i):
            Kc[j,j] += rho_w * cp_w * |Q|
            Kc[i,j] -= rho_w * cp_w * |Q|

    Parameters
    ----------
    nodes : list of Node
        All nodes in network.
    pipes : list of Pipe
        All pipes with flow rates.
    Q : numpy.ndarray
        Flow rates (m3/s), shape (n_pipes,).
    config : dict
        Configuration dictionary.

    Returns
    -------
    scipy.sparse.csr_matrix
        Convective matrix [Kc], shape (n_nodes, n_nodes).
        Note: this matrix is NOT symmetric!

    Example
    -------
    Kc = build_Kc_matrix(nodes, pipes, Q, config)
    print(f"Kc shape: {Kc.shape}")
    print(f"Symmetric: {False}  (upwind scheme)")
    """
    n_nodes = len(nodes)
    Kc      = lil_matrix((n_nodes, n_nodes), dtype=float)

    # get fluid properties from config
    thermal = config.get('grenier2018', {})
    rho_w   = thermal.get('rho_water', 1000.0)
    cp_w    = thermal.get('cp_water',  4182.0)

    n_convective = 0

    for idx, pipe in enumerate(pipes):
        i    = pipe.node_i
        j    = pipe.node_j
        q    = Q[idx]

        # heat capacity flux
        rho_cp_Q = rho_w * cp_w * q

        # upwind scheme
        if q > 0:
            # flow from i to j
            # upwind temperature = T_i
            Kc[i, i] += rho_cp_Q
            Kc[j, i] -= rho_cp_Q

        elif q < 0:
            # flow from j to i
            # upwind temperature = T_j
            Kc[j, j] -= rho_cp_Q
            Kc[i, j] += rho_cp_Q

        n_convective += 1

    print(f"  Assembled [Kc] matrix: {n_nodes}x{n_nodes}")
    print(f"  Non-zero entries     : {Kc.nnz}")
    print(f"  Convective pipes     : {n_convective}")

    return csr_matrix(Kc)


def compute_peclet_per_pipe(pipes, Q, config):
    """
    Compute Peclet number for each pipe.

    Useful for diagnosing advection vs conduction regime.

    Parameters
    ----------
    pipes : list of Pipe
    Q : numpy.ndarray
        Flow rates (m3/s).
    config : dict

    Returns
    -------
    numpy.ndarray
        Peclet numbers per pipe.
    """
    thermal   = config.get('grenier2018', {})
    rho_w     = thermal.get('rho_water',    1000.0)
    cp_w      = thermal.get('cp_water',     4182.0)
    lambda_w  = thermal.get('lambda_water', 0.6)

    Pe = np.zeros(len(pipes))
    for idx, pipe in enumerate(pipes):
        area     = pipe.aperture * pipe.width
        Pe[idx]  = compute_peclet_number(
            rho_w, cp_w, Q[idx],
            lambda_w, area, pipe.length
        )
    return Pe


def print_convection_summary(pipes, Q, Pe):
    """
    Print summary of convective transport.

    Parameters
    ----------
    pipes : list of Pipe
    Q : numpy.ndarray
    Pe : numpy.ndarray
    """
    print("=" * 50)
    print("  CONVECTION SUMMARY")
    print("=" * 50)
    print(f"  Number of pipes   : {len(pipes)}")
    print(f"  Max |Q|           : {np.abs(Q).max():.3e} m3/s")
    print(f"  Mean |Q|          : {np.abs(Q).mean():.3e} m3/s")

    Pe_finite = Pe[np.isfinite(Pe)]
    if len(Pe_finite) > 0:
        print(f"\n  PECLET NUMBERS:")
        print(f"    Min  : {Pe_finite.min():.2f}")
        print(f"    Max  : {Pe_finite.max():.2f}")
        print(f"    Mean : {Pe_finite.mean():.2f}")

        n_adv  = np.sum(Pe_finite > 10)
        n_cond = np.sum(Pe_finite < 0.1)
        n_mix  = len(Pe_finite) - n_adv - n_cond

        print(f"\n  REGIME:")
        print(f"    Advection dominated (Pe>10)  : {n_adv}")
        print(f"    Mixed regime (0.1<Pe<10)     : {n_mix}")
        print(f"    Conduction dominated (Pe<0.1): {n_cond}")

    print("=" * 50)