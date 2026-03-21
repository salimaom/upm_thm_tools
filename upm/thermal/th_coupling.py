"""
th_coupling.py
==============
Thermo-Hydraulic (T-H) iterative coupling for UPM pipe network.

Solver strategy:
    1. Picard iteration (robust, guaranteed convergence)
    2. Future: Newton-Raphson with Picard fallback
    3. Future: Automatic time step control

Picard scheme per time step:
    k=0: initial guess T^0 = T_old, Phi^0 = Phi_old

    LOOP k = 1, 2, ... max_iter:
        1. Compute S_ice(T^(k-1))
        2. Update apertures a(S_ice)
        3. Solve flow [K(T^(k-1))]{Phi^k} = {Qs}
        4. Compute Q^k from Phi^k
        5. Build [Kt] and [Kc(Q^k)]
        6. Add latent heat to [C]
        7. Solve heat [C/dt + Kt + Kc]{T^k} = {rhs}
        8. Convergence check:
               R_T   = max|T^k   - T^(k-1)|   < tol_T
               R_Phi = max|Phi^k - Phi^(k-1)| < tol_P
        9. If converged → advance time step
           If not      → set T^(k-1)=T^k, repeat

Works for:
    - 1D pipes (local DFN around excavation)
    - 2D triangular mesh on fault planes (regional)
    - 3D pipe networks (full model)
    All node and pipe objects are 3D by design.

Future upgrade path:
    - Newton-Raphson with analytical Jacobian
      J[1,1] = K(T)
      J[1,2] = dK/dT * Phi  (via dK/da * da/dSice * dSice/dT)
      J[2,1] = dKc/dPhi
      J[2,2] = C/dt + Kt + Kc + dC_latent/dT
    - Picard fallback if NR diverges
    - Automatic dt control (reduce if no convergence)

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Grenier et al. (2018) - Picard coupling, InterFrost
    Ren et al. (2017)     - UPM 3D pipe network
    Chen et al. (2018)    - T-H coupling in fractures
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, csr_matrix, issparse

from upm.flow.conductance import (
    assign_conductances,
    compute_effective_aperture
)
from upm.flow.matrix_builder import build_K_matrix, build_Qs_vector
from upm.flow.boundary import apply_simple_bc
from upm.flow.solver import (
    solve_pressure,
    compute_pipe_flow_rates,
    assign_pressures_to_nodes
)
from upm.thermal.conduction import (
    build_Kt_matrix,
    apply_thermal_dirichlet_bc
)
from upm.thermal.convection import build_Kc_matrix
from upm.freezing.ice_saturation import (
    assign_ice_saturation,
    compute_latent_heat_capacity
)


# ─────────────────────────────────────────────────────────────────
# TEMPERATURE SOLVER
# ─────────────────────────────────────────────────────────────────

def solve_temperature(Kt, Kc, C, Qt, T_old, dt):
    """
    Solve transient heat equation for one time step.

    Implicit backward Euler:
        [C/dt + Kt + Kc]{T_new} = {Qt} + [C/dt]{T_old}

    Works for any pipe network topology in 3D.

    Parameters
    ----------
    Kt : scipy.sparse.csr_matrix
        Thermal conduction matrix (n_nodes x n_nodes).
    Kc : scipy.sparse.csr_matrix
        Thermal convection matrix, upwind (n_nodes x n_nodes).
        Asymmetric — do NOT add to Kt before solving!
    C : numpy.ndarray
        Volumetric heat capacity vector (J/m3/K), shape (n_nodes,).
        Includes latent heat contribution when phase_change=True.
    Qt : numpy.ndarray
        Heat source/sink vector (W), shape (n_nodes,).
    T_old : numpy.ndarray
        Temperature at previous time step (K), shape (n_nodes,).
    dt : float
        Time step size (s).

    Returns
    -------
    numpy.ndarray
        Temperature at new time step (K), shape (n_nodes,).

    Raises
    ------
    ValueError
        If solver returns NaN or Inf.
    """
    # diagonal heat capacity matrix C/dt
    C_dt = diags(C / dt, format='csr')

    # system matrix: [C/dt + Kt + Kc]
    A = C_dt + Kt + Kc

    # right hand side: {Qt} + [C/dt]{T_old}
    rhs = Qt + C_dt.dot(T_old)

    # solve linear system
    T_new = spsolve(A, rhs)

    # validate
    if np.any(np.isnan(T_new)) or np.any(np.isinf(T_new)):
        raise ValueError(
            "Temperature solver returned NaN or Inf.\n"
            "Check thermal BCs and matrix assembly.\n"
            "Consider reducing time step dt."
        )

    return T_new


# ─────────────────────────────────────────────────────────────────
# STATE UPDATE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def assign_temperatures_to_nodes(nodes, T):
    """
    Assign solved temperatures back to node objects.

    Parameters
    ----------
    nodes : list of Node
        3D nodes in the pipe network.
    T : numpy.ndarray
        Temperature solution (K), shape (n_nodes,).

    Returns
    -------
    list of Node
        Nodes with updated temperature_fluid attribute.
    """
    for node in nodes:
        node.temperature_fluid = float(T[node.node_id])
    return nodes


def update_pipe_apertures_from_ice(pipes, S_ice_dict,
                                    apertures_original):
    """
    Update effective pipe apertures from ice saturation.

    For each pipe, averages ice saturation at both endpoints
    then applies cubic law reduction:
        a_eff = a_0 * (1 - S_ice_avg)^(1/3)

    Works for 3D pipes — only aperture changes,
    not pipe geometry or connectivity.

    Parameters
    ----------
    pipes : list of Pipe
        All pipes in 3D network.
    S_ice_dict : dict
        {node_id: S_ice} ice saturation per node.
    apertures_original : numpy.ndarray
        Original mechanical apertures (m), shape (n_pipes,).
        Must be stored before any ice modification!

    Returns
    -------
    list of Pipe
        Pipes with updated aperture attribute.
    """
    for idx, pipe in enumerate(pipes):
        S_i = S_ice_dict.get(pipe.node_i, 0.0)
        S_j = S_ice_dict.get(pipe.node_j, 0.0)
        S_ice_avg = (S_i + S_j) / 2.0

        pipe.aperture = compute_effective_aperture(
            apertures_original[idx], S_ice_avg
        )

    return pipes


def update_heat_capacity_with_latent(C, nodes, config):
    """
    Add latent heat contribution to heat capacity vector.

    Apparent heat capacity method:
        C_total[i] = C_sensible[i] + C_latent[i]
        C_latent[i] = -rho_ice * L * dS_ice/dT * V_node[i]

    This implicitly handles the Stefan condition
    (energy absorbed/released during phase change).

    Works for any node type in 3D network.

    Parameters
    ----------
    C : numpy.ndarray
        Sensible heat capacity vector (J/m3/K).
    nodes : list of Node
        Nodes with current temperatures.
    config : dict
        Configuration with freezing parameters.

    Returns
    -------
    numpy.ndarray
        Updated heat capacity including latent heat (J/m3/K).
    """
    thermal  = config.get('grenier2018', {})
    Tm       = thermal.get('T_freeze_K',   273.15)
    Swr      = thermal.get('Sw_residual',  0.05)
    W        = thermal.get('W_freezing_K', 0.5)
    rho_ice  = thermal.get('rho_ice',      920.0)
    L_fusion = thermal.get('L_fusion',     334000.0)

    C_total = C.copy()

    for node in nodes:
        T_node   = node.temperature_fluid
        C_latent = compute_latent_heat_capacity(
            T_node, Tm, Swr, W, rho_ice, L_fusion
        )
        C_total[node.node_id] += C_latent

    return C_total


# ─────────────────────────────────────────────────────────────────
# CONVERGENCE CHECK
# ─────────────────────────────────────────────────────────────────

def check_convergence(T_new, T_iter, Phi_new, Phi_iter,
                       tol_T=1e-6, tol_P=1.0):
    """
    Check Picard convergence for both temperature and pressure.

    Residuals:
        R_T   = max|T_new   - T_iter|
        R_Phi = max|Phi_new - Phi_iter|

    Parameters
    ----------
    T_new : numpy.ndarray
        Temperature at current Picard iteration (K).
    T_iter : numpy.ndarray
        Temperature at previous Picard iteration (K).
    Phi_new : numpy.ndarray
        Pressure at current Picard iteration (Pa).
    Phi_iter : numpy.ndarray
        Pressure at previous Picard iteration (Pa).
    tol_T : float
        Temperature convergence tolerance (K). Default 1e-6.
    tol_P : float
        Pressure convergence tolerance (Pa). Default 1.0.

    Returns
    -------
    bool
        True if converged.
    float
        Temperature residual R_T (K).
    float
        Pressure residual R_Phi (Pa).
    """
    R_T   = np.max(np.abs(T_new   - T_iter))
    R_Phi = np.max(np.abs(Phi_new - Phi_iter))

    converged = (R_T < tol_T) and (R_Phi < tol_P)

    return converged, R_T, R_Phi


# ─────────────────────────────────────────────────────────────────
# MAIN PICARD T-H COUPLING LOOP
# ─────────────────────────────────────────────────────────────────

def run_th_coupling(nodes, pipes, config,
                    inlet_nodes, outlet_nodes,
                    inlet_pressure, outlet_pressure,
                    inlet_temp, outlet_temp,
                    T_initial=None):
    """
    Run full T-H coupled simulation using Picard iterations.

    Designed for 3D pipe networks — nodes and pipes
    can have any 3D geometry. The solver only operates
    on the network topology (node IDs and pipe connectivity),
    not on specific coordinates.

    Future upgrade path:
        - Replace inner Picard loop with Newton-Raphson
        - Keep Picard as fallback
        - Add automatic dt control

    Parameters
    ----------
    nodes : list of Node
        All 3D nodes in pipe network.
    pipes : list of Pipe
        All 3D pipe segments.
    config : dict
        Configuration dictionary.
    inlet_nodes : list of int
        Node IDs with prescribed inlet pressure and temperature.
    outlet_nodes : list of int
        Node IDs with prescribed outlet pressure and temperature.
    inlet_pressure : float
        Inlet pressure BC (Pa).
    outlet_pressure : float
        Outlet pressure BC (Pa).
    inlet_temp : float
        Inlet temperature BC (K).
    outlet_temp : float
        Outlet temperature BC (K).
    T_initial : float or None
        Initial temperature field (K).
        If None: uses inlet_temp everywhere.

    Returns
    -------
    dict
        Simulation results:
        'T_history'        : list of T arrays per time step
        'Phi_history'      : list of Phi arrays per time step
        'Q_history'        : list of Q arrays per time step
        'S_ice_history'    : list of S_ice dicts per time step
        'times'            : list of simulation times (s)
        'convergence_hist' : list of (R_T, R_Phi) per step
        'nodes'            : final node states
        'pipes'            : final pipe states
    """

    # ── Get solver parameters ─────────────────────────────────────
    solver   = config.get('solver', {})
    n_steps  = solver.get('time_steps',     100)
    dt       = solver.get('dt_seconds',     60.0)
    tol_T    = solver.get('tolerance',      1e-6)
    tol_P    = solver.get('tolerance',      1.0)
    max_iter = solver.get('max_iterations', 20)

    # ── Get physics toggles ───────────────────────────────────────
    do_thermal  = config.get('physics', {}).get(
        'thermal', True)
    do_convect  = config.get('thermal', {}).get(
        'convection', True)
    do_freeze   = config.get('physics', {}).get(
        'phase_change', False)

    n_nodes = len(nodes)
    n_pipes = len(pipes)

    # ── Store original apertures ──────────────────────────────────
    # CRITICAL: must store BEFORE any ice modification
    apertures_original = np.array(
        [p.aperture for p in pipes], dtype=float
    )

    # ── Initialize temperature field ──────────────────────────────
    T_init = T_initial if T_initial is not None else inlet_temp
    T      = np.full(n_nodes, T_init, dtype=float)

    # enforce BC temperatures on initial field
    for nid in inlet_nodes:
        T[nid] = inlet_temp
    for nid in outlet_nodes:
        T[nid] = outlet_temp

    nodes = assign_temperatures_to_nodes(nodes, T)

    # ── Initialize pressure field ─────────────────────────────────
    # first flow solve to get initial Phi
    pipes = assign_conductances(pipes, config)
    K     = build_K_matrix(nodes, pipes)
    Qs    = build_Qs_vector(nodes)
    K, Qs = apply_simple_bc(
        K, Qs, nodes,
        inlet_nodes, outlet_nodes,
        inlet_pressure, outlet_pressure
    )
    Phi  = solve_pressure(K, Qs)
    Q    = compute_pipe_flow_rates(pipes, nodes, Phi)
    nodes = assign_pressures_to_nodes(nodes, Phi)

    # ── Results storage ───────────────────────────────────────────
    T_history        = [T.copy()]
    Phi_history      = [Phi.copy()]
    Q_history        = [Q.copy()]
    S_ice_history    = [{}]
    convergence_hist = []
    times            = [0.0]

    # ── Print simulation header ───────────────────────────────────
    print("=" * 55)
    print("  T-H PICARD COUPLING — 3D PIPE NETWORK")
    print("=" * 55)
    print(f"  Nodes          : {n_nodes}")
    print(f"  Pipes          : {n_pipes}")
    print(f"  Time steps     : {n_steps}")
    print(f"  dt             : {dt} s")
    print(f"  Total time     : {n_steps*dt/86400:.2f} days")
    print(f"  Thermal        : {do_thermal}")
    print(f"  Convection     : {do_convect}")
    print(f"  Phase change   : {do_freeze}")
    print(f"  Max Picard iter: {max_iter}")
    print(f"  Tol T          : {tol_T} K")
    print(f"  Tol P          : {tol_P} Pa")
    print("=" * 55)

    # ── Time loop ─────────────────────────────────────────────────
    for step in range(n_steps):

        T_old   = T.copy()
        Phi_old = Phi.copy()
        t_curr  = (step + 1) * dt

        converged    = False
        n_iter_used  = 0
        R_T_final    = np.inf
        R_Phi_final  = np.inf

        # ── Picard iteration loop ─────────────────────────────────
        for iteration in range(max_iter):

            # store previous Picard iterate
            T_iter   = T.copy()
            Phi_iter = Phi.copy()

            # ── Step 1: Ice saturation ────────────────────────────
            if do_freeze:
                S_ice_dict, S_w_dict = assign_ice_saturation(
                    nodes, config
                )
                pipes = update_pipe_apertures_from_ice(
                    pipes, S_ice_dict, apertures_original
                )
            else:
                S_ice_dict = {n.node_id: 0.0 for n in nodes}
                S_w_dict   = {n.node_id: 1.0 for n in nodes}

            # ── Step 2: Solve flow ────────────────────────────────
            pipes = assign_conductances(pipes, config)
            K     = build_K_matrix(nodes, pipes)
            Qs    = build_Qs_vector(nodes)
            K, Qs = apply_simple_bc(
                K, Qs, nodes,
                inlet_nodes, outlet_nodes,
                inlet_pressure, outlet_pressure
            )
            Phi_new = solve_pressure(K, Qs)
            Q_new   = compute_pipe_flow_rates(
                pipes, nodes, Phi_new
            )
            nodes = assign_pressures_to_nodes(nodes, Phi_new)

            if not do_thermal:
                Phi = Phi_new
                Q   = Q_new
                converged = True
                break

            # ── Step 3: Build thermal matrices ────────────────────
            Kt, C = build_Kt_matrix(
                nodes, pipes, config, S_w_dict
            )
            Qt = np.zeros(n_nodes, dtype=float)

            # apply thermal Dirichlet BCs
            for nid in inlet_nodes:
                Kt, Qt = apply_thermal_dirichlet_bc(
                    Kt, Qt, nid, inlet_temp
                )
            for nid in outlet_nodes:
                Kt, Qt = apply_thermal_dirichlet_bc(
                    Kt, Qt, nid, outlet_temp
                )

            # ── Step 4: Build convection matrix ───────────────────
            if do_convect:
                Kc = build_Kc_matrix(
                    nodes, pipes, Q_new, config
                )
            else:
                Kc = csr_matrix((n_nodes, n_nodes))

            # ── Step 5: Add latent heat to capacity ───────────────
            if do_freeze:
                C = update_heat_capacity_with_latent(
                    C, nodes, config
                )

            # ── Step 6: Solve temperature ─────────────────────────
            T_new = solve_temperature(
                Kt, Kc, C, Qt, T_old, dt
            )

            # ── Step 7: Update state ──────────────────────────────
            Phi = Phi_new
            Q   = Q_new
            T   = T_new
            nodes = assign_temperatures_to_nodes(nodes, T)

            # ── Step 8: Check convergence ─────────────────────────
            converged, R_T, R_Phi = check_convergence(
                T, T_iter, Phi, Phi_iter,
                tol_T, tol_P
            )

            n_iter_used = iteration + 1
            R_T_final   = R_T
            R_Phi_final = R_Phi

            if converged:
                break

        # ── Store results for this time step ──────────────────────
        T_history.append(T.copy())
        Phi_history.append(Phi.copy())
        Q_history.append(Q.copy())
        S_ice_history.append(S_ice_dict.copy())
        convergence_hist.append((R_T_final, R_Phi_final))
        times.append(t_curr)

        # ── Print progress ────────────────────────────────────────
        if (step + 1) % 10 == 0 or step == 0:
            T_C_min = T.min() - 273.15
            T_C_max = T.max() - 273.15
            conv_str = "OK" if converged else "NO"
            print(f"  Step {step+1:4d}/{n_steps} "
                  f"t={t_curr/86400:.3f}d "
                  f"T=[{T_C_min:.2f},{T_C_max:.2f}]C "
                  f"iter={n_iter_used:2d} "
                  f"R_T={R_T_final:.2e} "
                  f"conv={conv_str}")

    print("=" * 55)
    print("  Simulation complete!")
    print(f"  Final T: "
          f"{T.min()-273.15:.2f} to "
          f"{T.max()-273.15:.2f} C")
    print("=" * 55)

    return {
        'T_history'       : T_history,
        'Phi_history'     : Phi_history,
        'Q_history'       : Q_history,
        'S_ice_history'   : S_ice_history,
        'convergence_hist': convergence_hist,
        'times'           : times,
        'nodes'           : nodes,
        'pipes'           : pipes
    }


def print_th_summary(results):
    """
    Print summary of T-H simulation results.

    Parameters
    ----------
    results : dict
        Results from run_th_coupling.
    """
    times    = results['times']
    T_hist   = results['T_history']
    conv_hist = results['convergence_hist']

    print("=" * 55)
    print("  T-H SIMULATION SUMMARY")
    print("=" * 55)
    print(f"  Total time steps  : {len(times)-1}")
    print(f"  Total time        : {times[-1]/86400:.2f} days")

    if conv_hist:
        R_T_all = [r[0] for r in conv_hist]
        print(f"\n  CONVERGENCE:")
        print(f"    Max R_T  : {max(R_T_all):.2e} K")
        print(f"    Mean R_T : {np.mean(R_T_all):.2e} K")

    print(f"\n  INITIAL temperature:")
    print(f"    Min : {T_hist[0].min()-273.15:.2f} C")
    print(f"    Max : {T_hist[0].max()-273.15:.2f} C")

    print(f"\n  FINAL temperature:")
    print(f"    Min : {T_hist[-1].min()-273.15:.2f} C")
    print(f"    Max : {T_hist[-1].max()-273.15:.2f} C")
    print("=" * 55)