"""
conductance.py
==============
Hydraulic conductance calculations for UPM pipe network.

Cubic law for fracture flow:
    Q = (a³ · w) / (12 · mu · L) · ΔP
    K = (a³ · w) / (12 · mu · L)

where:
    a   = hydraulic aperture (m)
    w   = equivalent pipe width (m)
    mu  = dynamic viscosity (Pa·s)
    L   = pipe length (m)
    K   = conductance (m³/Pa·s)

Freezing effect on aperture:
    a_eff = a · (1 - S_ice)^omega
    where S_ice is ice saturation
    and omega is impedance parameter

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Ren et al. (2017) - Unified Pipe Network Method
    Chen et al. (2018) - T-H coupling in fractured rock
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────
# CORE CONDUCTANCE FUNCTION
# ─────────────────────────────────────────────────────────────────

def compute_conductance(aperture, width, length, viscosity):
    """
    Compute hydraulic conductance of a fracture pipe.

    Uses cubic law:
        K = a³ · w / (12 · mu · L)

    Parameters
    ----------
    aperture : float
        Hydraulic aperture a (m).
    width : float
        Equivalent pipe width w (m).
        For 1D local DFN pipes: w = aperture (unit width)
        For 2D fault zone pipes: w = Voronoi width from mesh
    length : float
        Pipe length L (m).
    viscosity : float
        Dynamic viscosity mu (Pa·s).
        Water at 20°C: 1e-3 Pa·s
        Water at 0°C:  1.79e-3 Pa·s

    Returns
    -------
    float
        Hydraulic conductance K (m³/Pa·s).

    Raises
    ------
    ValueError
        If any input is negative or zero.

    Example
    -------
    K = compute_conductance(
            aperture  = 0.001,
            width     = 0.001,
            length    = 5.0,
            viscosity = 1e-3
        )
    print(f"K = {K:.3e} m3/Pa/s")
    """
    # validate inputs
    if aperture <= 0:
        raise ValueError(f"Aperture must be positive, got {aperture}")
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")
    if viscosity <= 0:
        raise ValueError(f"Viscosity must be positive, got {viscosity}")

    return (aperture**3 * width) / (12.0 * viscosity * length)


def compute_conductance_array(apertures, widths, lengths, viscosity):
    """
    Compute hydraulic conductance for multiple pipes at once.

    Vectorized version of compute_conductance for efficiency.

    Parameters
    ----------
    apertures : numpy.ndarray
        Hydraulic apertures (m), shape (n_pipes,).
    widths : numpy.ndarray
        Equivalent pipe widths (m), shape (n_pipes,).
    lengths : numpy.ndarray
        Pipe lengths (m), shape (n_pipes,).
    viscosity : float
        Dynamic viscosity (Pa·s).

    Returns
    -------
    numpy.ndarray
        Hydraulic conductances (m³/Pa·s), shape (n_pipes,).

    Example
    -------
    apertures = np.array([0.001, 0.002, 0.0005])
    widths    = np.array([0.001, 0.002, 0.0005])
    lengths   = np.array([5.0,   3.0,   8.0])
    K = compute_conductance_array(apertures, widths, lengths, 1e-3)
    """
    apertures = np.asarray(apertures, dtype=float)
    widths    = np.asarray(widths,    dtype=float)
    lengths   = np.asarray(lengths,   dtype=float)

    return (apertures**3 * widths) / (12.0 * viscosity * lengths)


# ─────────────────────────────────────────────────────────────────
# VISCOSITY
# ─────────────────────────────────────────────────────────────────

def compute_viscosity_water(temperature_K):
    """
    Compute dynamic viscosity of water as function of temperature.

    Empirical formula from Chen et al. (2018):
        mu(T) = 1 / (29.83 * (T - 258.6))

    Valid range: 273 K to 373 K (0°C to 100°C)

    Parameters
    ----------
    temperature_K : float or numpy.ndarray
        Temperature in Kelvin.

    Returns
    -------
    float or numpy.ndarray
        Dynamic viscosity (Pa·s).

    Example
    -------
    mu_0C  = compute_viscosity_water(273.15)
    mu_20C = compute_viscosity_water(293.15)
    print(f"Viscosity at  0°C: {mu_0C:.4f} Pa·s")
    print(f"Viscosity at 20°C: {mu_20C:.4f} Pa·s")
    """
    return 1.0 / (29.83 * (temperature_K - 258.6))


# ─────────────────────────────────────────────────────────────────
# EFFECTIVE APERTURE WITH FREEZING
# ─────────────────────────────────────────────────────────────────

def compute_effective_aperture(aperture, ice_saturation, omega=10.0):
    """
    Compute effective hydraulic aperture accounting for ice.

    Hydraulic impedance function:
        a_eff = a · (1 - S_ice)^omega

    where omega is the impedance parameter (Hansson parameter).
    Higher omega → stronger reduction of flow by ice.

    Parameters
    ----------
    aperture : float or numpy.ndarray
        Mechanical hydraulic aperture (m).
    ice_saturation : float or numpy.ndarray
        Ice saturation S_ice (0 = no ice, 1 = fully frozen).
    omega : float
        Impedance parameter (default 10, from config).

    Returns
    -------
    float or numpy.ndarray
        Effective hydraulic aperture (m).

    Example
    -------
    a_eff = compute_effective_aperture(0.001, 0.5, omega=10)
    print(f"Effective aperture: {a_eff*1000:.4f} mm")
    """
    ice_saturation = np.clip(ice_saturation, 0.0, 1.0)
    return aperture * (1.0 - ice_saturation) ** omega


def compute_conductance_frozen(aperture, width, length,
                                viscosity, ice_saturation,
                                omega=10.0):
    """
    Compute hydraulic conductance with ice effect.

    Combines effective aperture and variable viscosity:
        a_eff = a · (1 - S_ice)^omega
        K = a_eff³ · w / (12 · mu · L)

    Parameters
    ----------
    aperture : float
        Mechanical hydraulic aperture (m).
    width : float
        Equivalent pipe width (m).
    length : float
        Pipe length (m).
    viscosity : float
        Dynamic viscosity (Pa·s).
    ice_saturation : float
        Ice saturation (0 to 1).
    omega : float
        Impedance parameter (default 10).

    Returns
    -------
    float
        Hydraulic conductance with ice effect (m³/Pa·s).

    Example
    -------
    K_frozen = compute_conductance_frozen(
        aperture=0.001, width=0.001, length=5.0,
        viscosity=1e-3, ice_saturation=0.3, omega=10
    )
    """
    a_eff = compute_effective_aperture(aperture, ice_saturation, omega)
    return compute_conductance(a_eff, width, length, viscosity)


# ─────────────────────────────────────────────────────────────────
# ASSIGN CONDUCTANCES TO PIPE NETWORK
# ─────────────────────────────────────────────────────────────────

def assign_conductances(pipes, config, ice_saturations=None):
    """
    Compute and assign hydraulic conductance to all pipes.

    Reads viscosity from config.
    If ice_saturations provided, uses frozen conductance.
    Otherwise uses standard cubic law.

    Parameters
    ----------
    pipes : list of Pipe
        Pipe objects from dfn_local.build_local_pipe_network().
    config : dict
        Configuration dictionary.
    ice_saturations : dict or None
        Optional dict mapping pipe_id to ice saturation.
        If None, assumes no ice (S_ice = 0).

    Returns
    -------
    list of Pipe
        Same pipes with conductance attribute set.

    Example
    -------
    pipes = assign_conductances(pipes, config)
    K_values = [p.conductance for p in pipes]
    """
    # get viscosity from config
    fluid = config.get('fluid', {}).get('water', {})
    mu    = fluid.get('viscosity', 1e-3)

    # get freezing params from config
    freezing = config.get('freezing', {})
    omega    = freezing.get('omega_impedance', 10.0)

    # check if freezing is active
    phase_change = config.get('physics', {}).get('phase_change', False)

    n_assigned = 0
    for pipe in pipes:

        # get ice saturation if available
        if phase_change and ice_saturations is not None:
            S_ice = ice_saturations.get(pipe.pipe_id, 0.0)
            pipe.conductance = compute_conductance_frozen(
                aperture       = pipe.aperture,
                width          = pipe.width,
                length         = pipe.length,
                viscosity      = mu,
                ice_saturation = S_ice,
                omega          = omega
            )
        else:
            pipe.conductance = compute_conductance(
                aperture  = pipe.aperture,
                width     = pipe.width,
                length    = pipe.length,
                viscosity = mu
            )
        n_assigned += 1

    print(f"  Assigned conductances to {n_assigned} pipes")
    print(f"  Viscosity used: {mu:.2e} Pa·s")
    if phase_change:
        print(f"  Freezing active: omega = {omega}")

    return pipes


def print_conductance_summary(pipes):
    """
    Print summary of conductance values.

    Parameters
    ----------
    pipes : list of Pipe
        Pipes with conductance assigned.
    """
    if not pipes:
        print("No pipes to summarize.")
        return

    K_values = [p.conductance for p in pipes]

    print("=" * 50)
    print("  CONDUCTANCE SUMMARY")
    print("=" * 50)
    print(f"  Number of pipes : {len(pipes)}")
    print(f"  K min  : {min(K_values):.3e} m³/Pa·s")
    print(f"  K max  : {max(K_values):.3e} m³/Pa·s")
    print(f"  K mean : {sum(K_values)/len(K_values):.3e} m³/Pa·s")
    print(f"\n  Ratio K_max/K_min : {max(K_values)/min(K_values):.1f}")
    print("=" * 50)
