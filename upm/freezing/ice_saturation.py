"""
ice_saturation.py
=================
Ice saturation as function of temperature for fracture network.

Freezing curve (McKenzie 2007, used in Grenier 2018):

    For T >= Tm:
        S_ice = 0
        S_w   = 1

    For T < Tm:
        S_ice = (1 - Swr) * [1 - exp(-((Tm - T) / W)^2)]
        S_w   = 1 - S_ice

where:
    Tm  = freezing point temperature (K)
    T   = current temperature (K)
    Swr = residual liquid water saturation (-)
    W   = freezing range parameter (K)
          controls sharpness of freeze-thaw transition
          W = 0.5 K for Grenier 2018 benchmark
          W calibrated from field data for Raglan

Effect on fracture aperture (cubic law):
    a_eff = a * (1 - S_ice)^(1/3)
    K_eff = K_0 * (1 - S_ice)

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    McKenzie et al. (2007) - freezing curve
    Grenier et al. (2018) - InterFrost benchmark
    Table 1: Swr=0.05, W=0.5K, Tm=273.15K
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────
# CORE FREEZING CURVE
# ─────────────────────────────────────────────────────────────────

def compute_ice_saturation(T, Tm=273.15, Swr=0.05, W=0.5):
    """
    Compute ice saturation from temperature.

    Freezing curve from McKenzie et al. (2007):
        For T >= Tm: S_ice = 0
        For T <  Tm: S_ice = (1-Swr) * [1 - exp(-((Tm-T)/W)^2)]

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (K).
    Tm : float
        Freezing point temperature (K). Default 273.15 K (0 degC).
    Swr : float
        Residual liquid water saturation (-).
        Grenier 2018: 0.05
        Fracture model: 0.0 (fractures can freeze completely)
    W : float
        Freezing range parameter (K).
        Controls sharpness of transition.
        Small W = sharp freeze (W=0.1)
        Large W = gradual freeze (W=1.0)
        Grenier 2018 benchmark: W=0.5

    Returns
    -------
    float or numpy.ndarray
        Ice saturation S_ice (0 to 1).

    Example
    -------
    S_ice = compute_ice_saturation(272.15, Tm=273.15, Swr=0.05, W=0.5)
    print(f"At -1 degC: S_ice = {S_ice:.3f}")
    """
    T    = np.asarray(T, dtype=float)
    dT   = Tm - T

    S_ice = np.where(
        T >= Tm,
        0.0,
        (1.0 - Swr) * (1.0 - np.exp(-(dT / W)**2))
    )

    return np.clip(S_ice, 0.0, 1.0)


def compute_water_saturation(T, Tm=273.15, Swr=0.05, W=0.5):
    """
    Compute liquid water saturation from temperature.

    S_w = 1 - S_ice

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (K).
    Tm : float
        Freezing point (K).
    Swr : float
        Residual liquid saturation.
    W : float
        Freezing range parameter (K).

    Returns
    -------
    float or numpy.ndarray
        Water saturation S_w (0 to 1).
    """
    return 1.0 - compute_ice_saturation(T, Tm, Swr, W)


def compute_dSice_dT(T, Tm=273.15, Swr=0.05, W=0.5):
    """
    Compute derivative of ice saturation with respect to temperature.

    dS_ice/dT = -2*(Tm-T)/W^2 * (1-Swr) * exp(-((Tm-T)/W)^2)

    Needed for:
        - Latent heat term in energy equation
        - Newton-Raphson linearization
        - Jacobian assembly

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (K).
    Tm : float
        Freezing point (K).
    Swr : float
        Residual liquid saturation.
    W : float
        Freezing range parameter (K).

    Returns
    -------
    float or numpy.ndarray
        dS_ice/dT (1/K). Always negative or zero.
    """
    T  = np.asarray(T, dtype=float)
    dT = Tm - T

    dSice_dT = np.where(
        T >= Tm,
        0.0,
        (2.0 * dT / W**2)
        * (1.0 - Swr)
        * np.exp(-(dT / W)**2)
        * (-1.0)    # negative because dT/dT = -1
    )

    return dSice_dT


# ─────────────────────────────────────────────────────────────────
# LATENT HEAT TERM
# ─────────────────────────────────────────────────────────────────

def compute_latent_heat_capacity(T, Tm=273.15, Swr=0.05, W=0.5,
                                  rho_ice=920.0, L_fusion=334000.0):
    """
    Compute apparent heat capacity due to latent heat of fusion.

    From Grenier et al. (2018) Eq. 2:
        C_latent = -rho_ice * L * dS_ice/dT

    This term is added to the heat capacity matrix [C]
    to account for energy absorbed/released during
    phase change.

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature (K).
    Tm : float
        Freezing point (K).
    Swr : float
        Residual liquid saturation.
    W : float
        Freezing range parameter (K).
    rho_ice : float
        Ice density (kg/m3). Default 920.
    L_fusion : float
        Latent heat of fusion (J/kg). Default 334000.

    Returns
    -------
    float or numpy.ndarray
        Apparent latent heat capacity (J/m3/K).
        Always positive (energy storage term).
    """
    dSice_dT = compute_dSice_dT(T, Tm, Swr, W)
    return -rho_ice * L_fusion * dSice_dT


# ─────────────────────────────────────────────────────────────────
# ASSIGN TO NODES
# ─────────────────────────────────────────────────────────────────

def assign_ice_saturation(nodes, config):
    """
    Compute and assign ice saturation to all nodes
    based on current nodal temperatures.

    Parameters
    ----------
    nodes : list of Node
        All nodes with temperature_fluid assigned.
    config : dict
        Configuration dictionary with freezing parameters.

    Returns
    -------
    dict
        {node_id: S_ice} for all nodes.
    dict
        {node_id: S_w} for all nodes.

    Example
    -------
    S_ice_dict, S_w_dict = assign_ice_saturation(nodes, config)
    """
    thermal = config.get('grenier2018', {})
    Tm  = thermal.get('T_freeze_K',   273.15)
    Swr = thermal.get('Sw_residual',  0.05)
    W   = thermal.get('W_freezing_K', 0.5)

    S_ice_dict = {}
    S_w_dict   = {}

    for node in nodes:
        T     = node.temperature_fluid
        S_ice = compute_ice_saturation(T, Tm, Swr, W)
        S_w   = 1.0 - S_ice

        S_ice_dict[node.node_id] = float(S_ice)
        S_w_dict[node.node_id]   = float(S_w)

    return S_ice_dict, S_w_dict


# ─────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────

def plot_freezing_curves(Tm=273.15, Swr=0.05,
                          W_values=None,
                          T_range_C=(-3.0, 0.5),
                          save_path=None):
    """
    Plot ice saturation curves for different W values.

    Reproduces your original Courbes_saturation_glace.py
    as a production-quality function.

    Parameters
    ----------
    Tm : float
        Freezing point (K).
    Swr : float
        Residual liquid water saturation.
    W_values : list or None
        List of W values to plot.
        Default: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    T_range_C : tuple
        Temperature range in Celsius (min, max).
    save_path : str or None
        Path to save figure. If None, just displays.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    fig = plot_freezing_curves(Swr=0.05, W_values=[0.3, 0.5, 1.0])
    """
    if W_values is None:
        W_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # temperature axis in Celsius then convert to K
    T_C   = np.linspace(T_range_C[0], T_range_C[1], 500)
    T_K   = T_C + 273.15
    Tm_T  = Tm - T_K   # (Tm - T) in K = degrees below freezing

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = cm.plasma(np.linspace(0.1, 0.85, len(W_values)))

    for W, color in zip(W_values, colors):
        S_ice = compute_ice_saturation(T_K, Tm, Swr, W)
        ax.plot(
            Tm_T, S_ice * 100,
            label=f'W = {W}',
            color=color,
            linewidth=2
        )

    # mark Grenier 2018 parameters
    ax.axvline(x=0.5, color='grey', linestyle='--',
               alpha=0.5, linewidth=1)
    ax.text(0.52, 50,
            'W = 0.5\nInterfrost\n(Grenier 2018)',
            fontsize=9, color='grey')

    ax.set_xlabel(r'$T_m - T$ (°C)', fontsize=13)
    ax.set_ylabel(r'$S_{glace}$ (%)', fontsize=13)
    ax.set_title(
        r'$S_{glace} = (1 - S_{sl,residuel}) '
        r'\cdot \left[1 - \exp\left(-\left('
        r'\frac{T_m - T}{W}\right)^2\right)\right]$'
        f'\n$S_{{sl,residuel}} = {Swr}$',
        fontsize=12
    )
    ax.set_xlim(max(0, T_range_C[0]), T_range_C[1])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0f}%')
    )
    ax.legend(title='Parametre W', fontsize=10,
              title_fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to: {save_path}")

    plt.show()
    return fig


def print_freezing_summary(T_values_C, Tm=273.15,
                            Swr=0.05, W=0.5,
                            rho_ice=920.0,
                            L_fusion=334000.0):
    """
    Print ice saturation table for key temperatures.

    Parameters
    ----------
    T_values_C : list
        Temperatures in Celsius to evaluate.
    Tm : float
        Freezing point (K).
    Swr : float
        Residual liquid saturation.
    W : float
        Freezing range parameter (K).
    rho_ice : float
        Ice density (kg/m3).
    L_fusion : float
        Latent heat of fusion (J/kg).
    """
    print("=" * 60)
    print(f"  FREEZING CURVE  Swr={Swr}  W={W}K")
    print("=" * 60)
    print(f"  {'T (C)':>8} {'T (K)':>8} {'S_ice':>8} "
          f"{'S_w':>8} {'C_latent':>14}")
    print(f"  {'':-<8} {'':-<8} {'':-<8} "
          f"{'':-<8} {'':-<14}")

    for T_C in T_values_C:
        T_K      = T_C + 273.15
        S_ice    = compute_ice_saturation(T_K, Tm, Swr, W)
        S_w      = 1.0 - S_ice
        C_latent = compute_latent_heat_capacity(
            T_K, Tm, Swr, W, rho_ice, L_fusion
        )
        print(f"  {T_C:>8.2f} {T_K:>8.2f} "
              f"{S_ice:>8.4f} {S_w:>8.4f} "
              f"{C_latent:>14.2e}")

    print("=" * 60)