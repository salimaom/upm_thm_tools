"""
boundary.py
===========
Boundary condition application for UPM pipe network solver.

Author: Salim Hammoum
Polytechnique Montreal - 2026
"""

import numpy as np
from scipy.sparse import lil_matrix


def apply_dirichlet_bc(K, Qs, node_id, prescribed_pressure,
                        large_number=1e30):
    """
    Apply Dirichlet (prescribed pressure) boundary condition
    at a single node using the large number method.

    Parameters
    ----------
    K : scipy.sparse matrix
        Global conductance matrix.
    Qs : numpy.ndarray
        Source vector.
    node_id : int
        Node index where BC is applied.
    prescribed_pressure : float
        Prescribed pressure value (Pa).
    large_number : float
        Large multiplier (default 1e30).

    Returns
    -------
    K : scipy.sparse matrix
        Modified conductance matrix.
    Qs : numpy.ndarray
        Modified source vector.
    """
    K_lil = lil_matrix(K)
    K_lil[node_id, node_id] *= large_number
    Qs[node_id] = K_lil[node_id, node_id] * prescribed_pressure
    return K_lil.tocsr(), Qs


def apply_neumann_bc(Qs, node_id, flux):
    """
    Apply Neumann (prescribed flux) boundary condition.

    Parameters
    ----------
    Qs : numpy.ndarray
        Source vector.
    node_id : int
        Node index where BC is applied.
    flux : float
        Prescribed flux (m³/s).
        Positive = inflow, Negative = outflow.

    Returns
    -------
    numpy.ndarray
        Modified source vector.
    """
    Qs[node_id] += flux
    return Qs


def apply_boundary_conditions(K, Qs, nodes, config,
                               piezometer_data=None):
    """
    Apply all boundary conditions from config.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Global conductance matrix.
    Qs : numpy.ndarray
        Source vector.
    nodes : list of Node
        All nodes in the network.
    config : dict
        Configuration dictionary.
    piezometer_data : pandas.DataFrame or None
        Piezometer pressure data with columns:
        node_id, pressure_pa

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Modified conductance matrix.
    Qs : numpy.ndarray
        Modified source vector.
    n_bc : int
        Number of boundary conditions applied.
    """
    bc_config = config.get('boundary_conditions', {})
    n_bc      = 0

    # fixed pressure nodes from config
    fixed_nodes = bc_config.get('fixed_pressure_nodes', [])
    for bc in fixed_nodes:
        node_id  = bc.get('node_id')
        pressure = bc.get('pressure_pa', 0.0)
        if node_id is not None and node_id < len(nodes):
            K, Qs = apply_dirichlet_bc(K, Qs, node_id, pressure)
            nodes[node_id].is_boundary    = True
            nodes[node_id].boundary_value = pressure
            n_bc += 1
            print(f"  BC applied: node {node_id} → "
                  f"P = {pressure:.2e} Pa")

    # piezometer data
    if piezometer_data is not None:
        for _, row in piezometer_data.iterrows():
            node_id  = int(row['node_id'])
            pressure = float(row['pressure_pa'])
            if node_id < len(nodes):
                K, Qs = apply_dirichlet_bc(
                    K, Qs, node_id, pressure
                )
                nodes[node_id].is_boundary    = True
                nodes[node_id].boundary_value = pressure
                n_bc += 1
                print(f"  Piezometer BC: node {node_id} → "
                      f"P = {pressure:.2e} Pa")

    # excavation nodes
    if config.get('physics', {}).get('excavation', False):
        excavation_nodes = bc_config.get('excavation_nodes', [])
        for node_id in excavation_nodes:
            if node_id < len(nodes):
                K, Qs = apply_dirichlet_bc(
                    K, Qs, node_id, 0.0
                )
                nodes[node_id].is_boundary    = True
                nodes[node_id].boundary_value = 0.0
                n_bc += 1
                print(f"  Excavation BC: node {node_id} → "
                      f"P = 0 Pa")

    print(f"  Total BCs applied: {n_bc}")
    return K, Qs, n_bc


def apply_simple_bc(K, Qs, nodes, inlet_nodes,
                    outlet_nodes, inlet_pressure,
                    outlet_pressure):
    """
    Apply simple inlet/outlet pressure boundary conditions.
    Useful for testing and validation.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Global conductance matrix.
    Qs : numpy.ndarray
        Source vector.
    nodes : list of Node
        All nodes in the network.
    inlet_nodes : list of int
        Node IDs with prescribed inlet pressure.
    outlet_nodes : list of int
        Node IDs with prescribed outlet pressure.
    inlet_pressure : float
        Inlet pressure (Pa).
    outlet_pressure : float
        Outlet pressure (Pa).

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Modified conductance matrix.
    Qs : numpy.ndarray
        Modified source vector.
    """
    n_bc = 0

    for node_id in inlet_nodes:
        K, Qs = apply_dirichlet_bc(
            K, Qs, node_id, inlet_pressure
        )
        nodes[node_id].is_boundary    = True
        nodes[node_id].boundary_value = inlet_pressure
        n_bc += 1

    for node_id in outlet_nodes:
        K, Qs = apply_dirichlet_bc(
            K, Qs, node_id, outlet_pressure
        )
        nodes[node_id].is_boundary    = True
        nodes[node_id].boundary_value = outlet_pressure
        n_bc += 1

    print(f"  Simple BC applied:")
    print(f"    Inlet  nodes {inlet_nodes}: "
          f"P = {inlet_pressure:.2e} Pa")
    print(f"    Outlet nodes {outlet_nodes}: "
          f"P = {outlet_pressure:.2e} Pa")
    print(f"    Total: {n_bc} BCs")

    return K, Qs


def check_bc_sufficient(nodes):
    """
    Check if enough boundary conditions are applied.
    Need at least one Dirichlet BC to solve.

    Parameters
    ----------
    nodes : list of Node

    Returns
    -------
    bool
        True if system has at least one BC.
    """
    n_bc = sum(1 for n in nodes if n.is_boundary)

    if n_bc == 0:
        print("  WARNING: No boundary conditions applied!")
        print("  System is singular — cannot solve.")
        print("  Apply at least one pressure BC.")
        return False

    print(f"  BC check: {n_bc} boundary nodes found ✓")
    return True