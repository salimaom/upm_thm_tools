"""
conduction.py
=============
Heat conduction assembly for UPM pipe network.

Heat conduction equation (transient):
    rho_c * dT/dt = div(lambda * grad(T)) + Q_exchange

Discretized for pipe network:
    [C] {dT/dt} + [Kt] {T} = {Qt}

where:
    [C]  = heat capacity matrix (diagonal)
    [Kt] = thermal conductance matrix
    {T}  = nodal temperature vector
    {Qt} = heat source/sink vector

Thermal conductance between nodes i and j:
    Kt(i,j) = lambda_eff * A / L

where:
    lambda_eff = effective thermal conductivity (W/m/K)
    A          = cross-sectional area (m2)
    L          = pipe length (m)

For fracture pipes:
    lambda_eff = Sw * lambda_water + (1-Sw) * lambda_ice
    A          = aperture * width

For matrix pipes (heat only):
    lambda_eff = eps*Sw*lw + eps*(1-Sw)*li + (1-eps)*ls
    A          = Voronoi cross section

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Grenier et al. (2018) - InterFrost benchmark
    Ren et al. (2017) - UPM method
    Chen et al. (2018) - T-H coupling
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ─────────────────────────────────────────────────────────────────
# THERMAL PROPERTIES
# ─────────────────────────────────────────────────────────────────

def compute_lambda_fracture(Sw, lambda_water=0.6, lambda_ice=2.14):
    """
    Compute effective thermal conductivity in fracture.

    Linear mixing between water and ice:
        lambda_eff = Sw * lambda_water + (1-Sw) * lambda_ice

    Parameters
    ----------
    Sw : float or numpy.ndarray
        Water saturation (0=fully frozen, 1=fully liquid).
    lambda_water : float
        Thermal conductivity of water (W/m/K). Default 0.6.
    lambda_ice : float
        Thermal conductivity of ice (W/m/K). Default 2.14.

    Returns
    -------
    float or numpy.ndarray
        Effective thermal conductivity (W/m/K).

    Example
    -------
    lam = compute_lambda_fracture(Sw=0.8)
    print(f"lambda_eff = {lam:.3f} W/m/K")
    """
    return Sw * lambda_water + (1.0 - Sw) * lambda_ice


def compute_lambda_matrix(Sw, porosity=0.37,
                           lambda_water=0.6,
                           lambda_ice=2.14,
                           lambda_solid=9.0):
    """
    Compute effective thermal conductivity of rock matrix.

    Bulk mixing law from Grenier et al. (2018):
        lambda_T = eps*Sw*lw + eps*(1-Sw)*li + (1-eps)*ls

    Parameters
    ----------
    Sw : float or numpy.ndarray
        Water saturation.
    porosity : float
        Rock matrix porosity. Default 0.37.
    lambda_water : float
        Thermal conductivity of water (W/m/K). Default 0.6.
    lambda_ice : float
        Thermal conductivity of ice (W/m/K). Default 2.14.
    lambda_solid : float
        Thermal conductivity of solid grains (W/m/K). Default 9.0.

    Returns
    -------
    float or numpy.ndarray
        Bulk thermal conductivity (W/m/K).
    """
    eps = porosity
    return (eps * Sw * lambda_water
            + eps * (1.0 - Sw) * lambda_ice
            + (1.0 - eps) * lambda_solid)


def compute_heat_capacity(Sw, porosity=0.37,
                           rho_water=1000.0, cp_water=4182.0,
                           rho_ice=920.0,   cp_ice=2060.0,
                           rho_solid=2650.0, cp_solid=835.0):
    """
    Compute volumetric heat capacity of rock matrix.

    From Grenier et al. (2018) Table 1:
        (rho_c)_T = eps*Sw*rho_w*cp_w
                  + eps*(1-Sw)*rho_i*cp_i
                  + (1-eps)*rho_s*cp_s

    Parameters
    ----------
    Sw : float or numpy.ndarray
        Water saturation.
    porosity : float
        Rock matrix porosity.
    rho_water, cp_water : float
        Water density (kg/m3) and specific heat (J/kg/K).
    rho_ice, cp_ice : float
        Ice density and specific heat.
    rho_solid, cp_solid : float
        Solid grain density and specific heat.

    Returns
    -------
    float or numpy.ndarray
        Volumetric heat capacity (J/m3/K).
    """
    eps = porosity
    return (eps * Sw * rho_water * cp_water
            + eps * (1.0 - Sw) * rho_ice * cp_ice
            + (1.0 - eps) * rho_solid * cp_solid)


def compute_heat_capacity_fracture(Sw,
                                    rho_water=1000.0, cp_water=4182.0,
                                    rho_ice=920.0,   cp_ice=2060.0):
    """
    Compute volumetric heat capacity inside a fracture.

    Only fluid phases (no solid):
        (rho_c)_f = Sw*rho_w*cp_w + (1-Sw)*rho_i*cp_i

    Parameters
    ----------
    Sw : float or numpy.ndarray
        Water saturation.
    rho_water, cp_water : float
        Water density and specific heat.
    rho_ice, cp_ice : float
        Ice density and specific heat.

    Returns
    -------
    float or numpy.ndarray
        Volumetric heat capacity (J/m3/K).
    """
    return (Sw * rho_water * cp_water
            + (1.0 - Sw) * rho_ice * cp_ice)


# ─────────────────────────────────────────────────────────────────
# THERMAL CONDUCTANCE MATRIX
# ─────────────────────────────────────────────────────────────────

def compute_thermal_conductance(lambda_eff, area, length):
    """
    Compute thermal conductance of a pipe.

    Kt = lambda_eff * A / L

    Parameters
    ----------
    lambda_eff : float
        Effective thermal conductivity (W/m/K).
    area : float
        Cross-sectional area (m2).
        For fracture pipes: aperture * width
    length : float
        Pipe length (m).

    Returns
    -------
    float
        Thermal conductance (W/K).
    """
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")
    return lambda_eff * area / length


def build_Kt_matrix(nodes, pipes, config,
                    Sw_values=None):
    """
    Assemble global thermal conductance matrix [Kt].

    Same assembly rules as hydraulic K matrix:
        Kt[i,i] += Kt_pipe
        Kt[i,j] -= Kt_pipe
        Kt[j,i] -= Kt_pipe
        Kt[j,j] += Kt_pipe

    Parameters
    ----------
    nodes : list of Node
        All nodes in network.
    pipes : list of Pipe
        All pipes in network.
    config : dict
        Configuration dictionary.
    Sw_values : dict or None
        Water saturation per node {node_id: Sw}.
        If None, assumes fully liquid (Sw=1).

    Returns
    -------
    scipy.sparse.csr_matrix
        Global thermal conductance matrix [Kt].
    numpy.ndarray
        Heat capacity vector {C}, shape (n_nodes,).

    Example
    -------
    Kt, C = build_Kt_matrix(nodes, pipes, config)
    print(f"Kt shape: {Kt.shape}")
    """
    n_nodes = len(nodes)
    Kt      = lil_matrix((n_nodes, n_nodes), dtype=float)
    C       = np.zeros(n_nodes, dtype=float)

    # get thermal properties from config
    thermal  = config.get('grenier2018', {})
    lam_w    = thermal.get('lambda_water', 0.6)
    lam_i    = thermal.get('lambda_ice',   2.14)
    lam_s    = thermal.get('lambda_solid', 9.0)
    rho_w    = thermal.get('rho_water',    1000.0)
    cp_w     = thermal.get('cp_water',     4182.0)
    rho_i    = thermal.get('rho_ice',      920.0)
    cp_i     = thermal.get('cp_ice',       2060.0)
    rho_s    = thermal.get('rho_solid',    2650.0)
    cp_s     = thermal.get('cp_solid',     835.0)
    eps      = thermal.get('porosity',     0.37)

    for pipe in pipes:
        i = pipe.node_i
        j = pipe.node_j

        # get water saturation at pipe midpoint
        Sw_i = Sw_values.get(i, 1.0) if Sw_values else 1.0
        Sw_j = Sw_values.get(j, 1.0) if Sw_values else 1.0
        Sw   = (Sw_i + Sw_j) / 2.0

        # thermal conductivity based on pipe type
        if pipe.pipe_type == 'fracture':
            lam_eff = compute_lambda_fracture(Sw, lam_w, lam_i)
            area    = pipe.aperture * pipe.width

        elif pipe.pipe_type == 'matrix':
            lam_eff = compute_lambda_matrix(
                Sw, eps, lam_w, lam_i, lam_s
            )
            area    = pipe.width * pipe.aperture

        else:
            lam_eff = lam_w
            area    = pipe.aperture * pipe.width

        # thermal conductance
        Kt_pipe = compute_thermal_conductance(
            lam_eff, area, pipe.length
        )

        # assemble into matrix
        Kt[i, i] += Kt_pipe
        Kt[j, j] += Kt_pipe
        Kt[i, j] -= Kt_pipe
        Kt[j, i] -= Kt_pipe

    # build heat capacity vector
    for node in nodes:
        Sw_node = Sw_values.get(node.node_id, 1.0) \
                  if Sw_values else 1.0

        if node.node_type == 'fracture_center':
            rho_c = compute_heat_capacity_fracture(
                Sw_node, rho_w, cp_w, rho_i, cp_i
            )
            # volume = aperture * width * length / 2
            # approximated as small control volume
            vol = 1.0   # placeholder until Voronoi volumes

        elif node.node_type == 'intersection':
            rho_c = compute_heat_capacity_fracture(
                Sw_node, rho_w, cp_w, rho_i, cp_i
            )
            vol = 1.0

        else:
            rho_c = compute_heat_capacity(
                Sw_node, eps,
                rho_w, cp_w,
                rho_i, cp_i,
                rho_s, cp_s
            )
            vol = 1.0

        C[node.node_id] = rho_c * vol

    print(f"  Assembled [Kt] matrix: {n_nodes}x{n_nodes}")
    print(f"  Non-zero entries: {Kt.nnz}")

    return csr_matrix(Kt), C


# ─────────────────────────────────────────────────────────────────
# THERMAL BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────

def apply_thermal_dirichlet_bc(Kt, Qt, node_id,
                                prescribed_temp,
                                large_number=1e30):
    """
    Apply Dirichlet (prescribed temperature) BC.

    Same large number method as hydraulic BC:
        Kt[i,i] *= large_number
        Qt[i]   = Kt[i,i] * prescribed_temp

    Parameters
    ----------
    Kt : scipy.sparse matrix
        Thermal conductance matrix.
    Qt : numpy.ndarray
        Heat source vector.
    node_id : int
        Node where BC is applied.
    prescribed_temp : float
        Prescribed temperature (K).
    large_number : float
        Large multiplier.

    Returns
    -------
    Kt, Qt : modified matrix and vector.
    """
    from scipy.sparse import lil_matrix
    Kt_lil = lil_matrix(Kt)
    Kt_lil[node_id, node_id] *= large_number
    Qt[node_id] = Kt_lil[node_id, node_id] * prescribed_temp
    return Kt_lil.tocsr(), Qt


def apply_thermal_simple_bc(Kt, Qt, nodes,
                             inlet_nodes, outlet_nodes,
                             T_inlet, T_outlet):
    """
    Apply simple inlet/outlet temperature BCs.

    Parameters
    ----------
    Kt : scipy.sparse matrix
    Qt : numpy.ndarray
    nodes : list of Node
    inlet_nodes : list of int
    outlet_nodes : list of int
    T_inlet : float
        Inlet temperature (K).
    T_outlet : float
        Outlet temperature (K).

    Returns
    -------
    Kt, Qt : modified matrix and vector.
    """
    for node_id in inlet_nodes:
        Kt, Qt = apply_thermal_dirichlet_bc(
            Kt, Qt, node_id, T_inlet
        )
        nodes[node_id].temperature_fluid = T_inlet

    for node_id in outlet_nodes:
        Kt, Qt = apply_thermal_dirichlet_bc(
            Kt, Qt, node_id, T_outlet
        )
        nodes[node_id].temperature_fluid = T_outlet

    print(f"  Thermal BC applied:")
    print(f"    Inlet  nodes {inlet_nodes}: "
          f"T = {T_inlet - 273.15:.1f} C")
    print(f"    Outlet nodes {outlet_nodes}: "
          f"T = {T_outlet - 273.15:.1f} C")

    return Kt, Qt


def print_thermal_summary(Kt, C):
    """
    Print summary of thermal matrix properties.

    Parameters
    ----------
    Kt : scipy.sparse matrix
    C : numpy.ndarray
    """
    print("=" * 50)
    print("  THERMAL MATRIX SUMMARY")
    print("=" * 50)
    print(f"  Shape           : {Kt.shape}")
    print(f"  Non-zeros       : {Kt.nnz}")
    Kt_dense = Kt.toarray()
    diag = np.diag(Kt_dense)
    print(f"  Diag range      : "
          f"{diag.min():.3e} to {diag.max():.3e} W/K")
    print(f"  Heat capacity   : "
          f"{C.min():.3e} to {C.max():.3e} J/m3/K")
    print("=" * 50)