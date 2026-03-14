"""
config.py
=========
Configuration loader for UPM-THM Tools.
Reads JSON config files and makes parameters available to all modules.

Author: Salim Hammoum
Polytechnique Montreal - 2026
"""

import json
import os


def load_config(simulation_config_path):
    """
    Load simulation configuration by merging three levels:
    1. Master properties (default material properties)
    2. Site properties (site-specific overrides)
    3. Simulation config (run-specific parameters + toggles)

    Parameters
    ----------
    simulation_config_path : str
        Path to the simulation-specific JSON config file.

    Returns
    -------
    dict
        Merged configuration dictionary.

    Example
    -------
    config = load_config('notebooks/01_article1_TH/config_benchmark.json')
    permeability = config['rock_types']['granite']['permeability_m2']
    run_thermal  = config['physics']['thermal']
    """

    # load simulation config
    sim_config = _load_json(simulation_config_path)

    # load master properties if specified
    master_path = sim_config.get('master_properties', None)
    if master_path:
        master_path = _resolve_path(simulation_config_path, master_path)
        master = _load_json(master_path)
    else:
        master = {}

    # load site properties if specified
    site_path = sim_config.get('site_properties', None)
    if site_path:
        site_path = _resolve_path(simulation_config_path, site_path)
        site = _load_json(site_path)
    else:
        site = {}

    # merge all three levels
    config = {}
    config.update(master)
    config = _deep_merge(config, site.get('overrides', {}))
    config = _deep_merge(config, sim_config)

    # add resolved simulation directory
    config['_sim_dir'] = os.path.dirname(
                         os.path.abspath(simulation_config_path))

    return config


def get_fracture_properties(config, fracture_type):
    """
    Get properties for a specific fracture type.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    fracture_type : str
        Type of fracture: 'fault_zone', 'joint_family_1',
        'joint_family_2', 'random_joint'

    Returns
    -------
    dict
        Properties for the requested fracture type.

    Example
    -------
    props = get_fracture_properties(config, 'fault_zone')
    aperture = props['aperture_m']
    """
    fracture_types = config.get('fracture_types', {})
    if fracture_type not in fracture_types:
        available = list(fracture_types.keys())
        raise ValueError(
            f"Fracture type '{fracture_type}' not found. "
            f"Available types: {available}"
        )
    return fracture_types[fracture_type]


def get_rock_properties(config, rock_type):
    """
    Get properties for a specific rock type.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    rock_type : str
        Type of rock: 'granite', 'gabbro', 'basalt'

    Returns
    -------
    dict
        Properties for the requested rock type.

    Example
    -------
    props = get_rock_properties(config, 'gabbro')
    porosity = props['porosity']
    """
    rock_types = config.get('rock_types', {})
    if rock_type not in rock_types:
        available = list(rock_types.keys())
        raise ValueError(
            f"Rock type '{rock_type}' not found. "
            f"Available types: {available}"
        )
    return rock_types[rock_type]


def get_fluid_properties(config, fluid_type='water'):
    """
    Get properties for a specific fluid type.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    fluid_type : str
        Type of fluid: 'water' or 'ice'

    Returns
    -------
    dict
        Properties for the requested fluid type.
    """
    fluids = config.get('fluid', {})
    if fluid_type not in fluids:
        raise ValueError(f"Fluid type '{fluid_type}' not found.")
    return fluids[fluid_type]


def is_module_active(config, module_name):
    """
    Check if a physics module is active in the config.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    module_name : str
        Module name: 'flow', 'thermal', 'phase_change',
        'mechanical', 'excavation'

    Returns
    -------
    bool
        True if module is active, False otherwise.

    Example
    -------
    if is_module_active(config, 'thermal'):
        run_thermal_solver()
    """
    physics = config.get('physics', {})
    return physics.get(module_name, False)


def is_freezing_option_active(config, option_name):
    """
    Check if a specific freezing option is active.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    option_name : str
        Option name: 'ice_saturation', 'clapeyron',
        'impedance_function', 'aperture_update',
        'ice_jacking_check', 'cryo_suction'

    Returns
    -------
    bool
        True if option is active, False otherwise.
    """
    freezing = config.get('freezing_options', {})
    return freezing.get(option_name, False)


def get_solver_params(config):
    """
    Get solver parameters from config.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    dict
        Solver parameters.
    """
    return config.get('solver', {
        'transient': True,
        'iterative_thm': False,
        'max_iterations': 50,
        'tolerance': 1e-6,
        'time_steps': 100,
        'dt_days': 1.0,
        'implicit_scheme': True
    })


def get_output_dir(config):
    """
    Get the output directory for results.
    Creates it if it does not exist.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    str
        Path to output directory.
    """
    output = config.get('output', {})
    output_dir = output.get('output_dir', 'results/')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_config_summary(config):
    """
    Print a readable summary of active modules and key parameters.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    """
    print("=" * 50)
    print(f"  SIMULATION: {config.get('simulation_name', 'unnamed')}")
    print("=" * 50)

    print("\n  PHYSICS MODULES:")
    physics = config.get('physics', {})
    for module, active in physics.items():
        status = "ON  ✓" if active else "OFF ✗"
        print(f"    {module:<20} {status}")

    print("\n  FREEZING OPTIONS:")
    freezing = config.get('freezing_options', {})
    for option, active in freezing.items():
        status = "ON  ✓" if active else "OFF ✗"
        print(f"    {option:<25} {status}")

    print("\n  SOLVER:")
    solver = get_solver_params(config)
    print(f"    time_steps    : {solver.get('time_steps')}")
    print(f"    dt_days       : {solver.get('dt_days')}")
    print(f"    tolerance     : {solver.get('tolerance')}")
    print(f"    transient     : {solver.get('transient')}")
    print("=" * 50)


# private helper functions

def _load_json(path):
    """Load a JSON file and return as dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_path(base_path, relative_path):
    """Resolve a relative path from a base file location."""
    base_dir = os.path.dirname(os.path.abspath(base_path))
    return os.path.normpath(os.path.join(base_dir, relative_path))


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    Override values take priority.
    """
    result = base.copy()
    for key, value in override.items():
        if (key in result and
                isinstance(result[key], dict) and
                isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result