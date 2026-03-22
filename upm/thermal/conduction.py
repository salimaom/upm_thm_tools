"""
upm/thermal/conduction.py
=========================
Thermal conductance matrix for UPM-THM model.

Handles:
    - Thermal conductivity of fracture fluid (water/ice mix)
    - Thermal conductivity of rock matrix (solid/water/ice mix)
    - Heat capacity of porous medium
    - Assembly of thermal stiffness matrix [Kt]
    - Dirichlet thermal boundary conditions

Author: Salim Hammoum
Polytechnique Montreal - 2026
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ─────────────────────────────────────────────────────────────────
# THERMAL CONDUCTIVITY
# ─────────────────────────────────────────────────────────────────

def compute_lambda_fracture(Sw,
                             lambda_water=0.6,
                             lambda_ice=2.14):
    """
    Effective thermal conductivity of fracture fluid (water/ice mix).

    Linear mixing rule:
        lambda_f = Sw * lambda_w + (1 - Sw) * lambda_i

    Parameters
    ----------
    Sw : float
        Water saturation (0 to 1).
    lambda_water : float
        Thermal conductivity of water (W/m/K). Default 0.6.
    lambda_ice : float
        Thermal conductivity of ice (W/m/K). Default 2.14.

    Returns
    -------
    float : Effective thermal conductivity (W/m/K).
    """
    Sw = float(np.clip(Sw, 0.0, 1.0))
    return Sw * lambda_water + (1.0 - Sw) * lambda_ice


def compute_lambda_matrix(Sw,
                           porosity=0.37,
                           lambda_water=0.6,
                           lambda_ice=2.14,
                           lambda_solid=9.0):
    """
    Effective thermal conductivity of porous matrix (solid/water/ice).

    Linear mixing rule:
        lambda = eps*Sw*lw + eps*(1-Sw)*li + (1-eps)*ls

    Parameters
    ----------
    Sw : float
        Water saturation (0 to 1).
    porosity : float
        Porosity (0 to 1). Default 0.37.
    lambda_water : float
        Thermal conductivity of water (W/m/K). Default 0.6.
    lambda_ice : float
        Thermal conductivity of ice (W/m/K). Default 2.14.
    lambda_solid : float
        Thermal conductivity of solid (W/m/K). Default 9.0.

    Returns
    -------
    float : Effective thermal conductivity (W/m/K).
    """
    Sw  = float(np.clip(Sw, 0.0, 1.0))
    eps = float(np.clip(porosity, 0.0, 1.0))

    return (eps * Sw * lambda_water
            + eps * (1.0 - Sw) * lambda_ice
            + (1.0 - eps) * lambda_solid)


# ─────────────────────────────────────────────────────────────────
# HEAT CAPACITY
# ─────────────────────────────────────────────────────────────────

def compute_heat_capacity(Sw,
                           porosity=0.37,
                           rho_water=1000.0,
                           cp_water=4182.0,
                           rho_ice=920.0,
                           cp_ice=2060.0,
                           rho_solid=2650.0,
                           cp_solid=835.0):
    """
    Volumetric heat capacity of porous medium (J/m3/K).

    (rho*c)_T = eps*Sw*rho_w*cp_w
              + eps*(1-Sw)*rho_i*cp_i
              + (1-eps)*rho_s*cp_s

    Parameters
    ----------
    Sw : float
        Water saturation (0 to 1).
    porosity : float
    rho_water, cp_water : float
    rho_ice, cp_ice : float
    rho_solid, cp_solid : float

    Returns
    -------
    float : Volumetric heat capacity (J/m3/K).
    """
    Sw  = float(np.clip(Sw, 0.0, 1.0))
    eps = float(np.clip(porosity, 0.0, 1.0))

    return (eps * Sw * rho_water * cp_water
            + eps * (1.0 - Sw) * rho_ice * cp_ice
            + (1.0 - eps) * rho_solid * cp_solid)


def compute_heat_capacity_fracture(Sw,
                                    aperture,
                                    rho_water=1000.0,
                                    cp_water=4182.0,
                                    rho_ice=920.0,
                                    cp_ice=2060.0):
    """
    Volumetric heat capacity of fracture fluid (J/m3/K).

    (rho*c)_f = Sw*rho_w*cp_w + (1-Sw)*rho_i*cp_i

    Parameters
    ----------
    Sw : float
        Water saturation.
    aperture : float
        Fracture aperture (m) — not used in formula but
        kept for API consistency.

    Returns
    -------
    float : Volumetric heat capacity (J/m3/K).
    """
    Sw = float(np.clip(Sw, 0.0, 1.0))
    return Sw * rho_water * cp_water + (1.0 - Sw) * rho_ice * cp_ice


# ─────────────────────────────────────────────────────────────────
# THERMAL CONDUCTANCE MATRIX
# ─────────────────────────────────────────────────────────────────

def compute_thermal_conductance(pipe,
                                 lambda_eff,
                                 use_face_area=True):
    """
    Thermal conductance of a single pipe (W/K).

    Kt_pipe = lambda_eff * face_area / length

    Parameters
    ----------
    pipe : Pipe or Pipe2D object
        Must have .length, .face_area or .width, .aperture attributes.
    lambda_eff : float
        Effective thermal conductivity (W/m/K).
    use_face_area : bool
        If True use pipe.face_area, else compute from
        pipe.width * pipe.aperture.

    Returns
    -------
    float : Thermal conductance (W/K).
    """
    if use_face_area and hasattr(pipe, 'face_area'):
        area = pipe.face_area
    else:
        area = pipe.width * pipe.aperture

    return lambda_eff * area / pipe.length


def build_Kt_matrix(nodes, pipes,
                     Sw_dict=None,
                     porosity=0.37,
                     lambda_water=0.6,
                     lambda_ice=2.14,
                     lambda_solid=9.0,
                     use_matrix=True):
    """
    Assemble thermal conductance matrix [Kt].

    Standard FEM assembly:
        Kt[i,i] += kt
        Kt[j,j] += kt
        Kt[i,j] -= kt
        Kt[j,i] -= kt

    Parameters
    ----------
    nodes : list of Node objects
    pipes : list of Pipe objects
    Sw_dict : dict or None
        {node_id: Sw} water saturation per node.
        If None, assumes fully liquid (Sw=1).
    porosity : float
    lambda_water, lambda_ice, lambda_solid : float
    use_matrix : bool
        If True use matrix formula (porous medium).
        If False use fracture formula.

    Returns
    -------
    scipy.sparse.csr_matrix : [Kt] matrix (n_nodes x n_nodes)
    """
    n_nodes = len(nodes)
    Kt      = lil_matrix((n_nodes, n_nodes))

    for pipe in pipes:
        i = pipe.node_i
        j = pipe.node_j

        if Sw_dict is not None:
            Sw_i = Sw_dict.get(i, 1.0)
            Sw_j = Sw_dict.get(j, 1.0)
            Sw   = (Sw_i + Sw_j) / 2.0
        else:
            Sw = 1.0

        if use_matrix:
            lam = compute_lambda_matrix(
                Sw, porosity,
                lambda_water, lambda_ice, lambda_solid
            )
        else:
            lam = compute_lambda_fracture(
                Sw, lambda_water, lambda_ice
            )

        kt = compute_thermal_conductance(pipe, lam)

        Kt[i, i] += kt
        Kt[j, j] += kt
        Kt[i, j] -= kt
        Kt[j, i] -= kt

    return csr_matrix(Kt)


# ─────────────────────────────────────────────────────────────────
# BOUNDARY CONDITIONS
# ─────────────────────────────────────────────────────────────────

def apply_thermal_dirichlet_bc(Kt, Qt, node_id,
                                prescribed_temp,
                                large_number=1e30):
    """
    Apply Dirichlet (prescribed temperature) BC.

    FIX: sets diagonal directly to large_number instead of
    multiplying — avoids zero-diagonal bug when Kt[i,i] = 0.

        Kt[i,i]  = large_number
        Qt[i]    = large_number * prescribed_temp

    Parameters
    ----------
    Kt : scipy.sparse matrix
        Thermal conductance matrix.
    Qt : numpy.ndarray
        Thermal load vector.
    node_id : int
        Node index to apply BC.
    prescribed_temp : float
        Prescribed temperature (K).
    large_number : float
        Penalty number. Default 1e30.

    Returns
    -------
    Kt : scipy.sparse.csr_matrix
    Qt : numpy.ndarray
    """
    Kt_lil = Kt.tolil()

    # SET diagonal directly — do NOT multiply
    # (multiplying fails silently when diagonal is 0)
    Kt_lil[node_id, node_id] = large_number
    Qt[node_id]               = large_number * prescribed_temp

    return Kt_lil.tocsr(), Qt


def apply_thermal_simple_bc(Kt, Qt, nodes,
                              inlet_node_ids,
                              inlet_temp,
                              outlet_node_ids=None):
    """
    Apply simple inlet/outlet thermal BCs for testing.

    Parameters
    ----------
    Kt : scipy.sparse matrix
    Qt : numpy.ndarray
    nodes : list of Node objects
    inlet_node_ids : list of int
    inlet_temp : float (K)
    outlet_node_ids : list of int or None

    Returns
    -------
    Kt : scipy.sparse.csr_matrix
    Qt : numpy.ndarray
    """
    for nid in inlet_node_ids:
        Kt, Qt = apply_thermal_dirichlet_bc(
            Kt, Qt, nid, inlet_temp
        )

    return Kt, Qt


# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────

def print_thermal_summary(Kt, Qt, nodes, label="Thermal"):
    """
    Print summary of thermal matrix and load vector.

    Parameters
    ----------
    Kt : scipy.sparse matrix
    Qt : numpy.ndarray
    nodes : list of Node objects
    label : str
    """
    print(f"\n  [{label}] Kt matrix: "
          f"{Kt.shape[0]}x{Kt.shape[1]}")
    print(f"  Non-zeros : {Kt.nnz}")
    print(f"  Qt range  : "
          f"{Qt.min():.3e} to {Qt.max():.3e}")
    print(f"  Diagonal  : "
          f"min={Kt.diagonal().min():.3e}  "
          f"max={Kt.diagonal().max():.3e}")