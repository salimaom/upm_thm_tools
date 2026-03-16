"""
readers.py
==========
Data readers for UPM-THM Tools.
Reads fracture CSV, DXF files, and piezometer data.

Geological convention:
    dip_direction : azimuth of dip vector (0=North, 90=East)
    dip           : angle from horizontal (0=flat, 90=vertical)

Author: Salim Hammoum
Polytechnique Montreal - 2026
"""

import os
import numpy as np
import pandas as pd


def read_fracture_csv(csv_path, config=None):
    """
    Read fracture data from CSV file.
    Supports flexible column naming (French or English).

    Expected columns (minimum required):
        x, y, z        - fracture center coordinates (m)
        dip_direction  - azimuth of dip vector in degrees
                         (0=North, 90=East, 180=South, 270=West)
        dip            - dip angle in degrees
                         (0=horizontal, 90=vertical)
        longueur       - fracture length (m)

    Optional columns:
        aperture       - hydraulic aperture (m)
        fracture_type  - type identifier for config lookup
        width          - equivalent pipe width (m)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    config : dict, optional
        Configuration dictionary for default values.

    Returns
    -------
    pandas.DataFrame
        Cleaned fracture dataframe with standardized columns.

    Example
    -------
    df = read_fracture_csv('data/structural/fractures.csv', config)
    print(df.head())
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # load CSV
    df = pd.read_csv(csv_path)

    # standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # map French/English column names to standard names
    column_mapping = {
        # coordinates
        'x_c': 'x', 'xc': 'x', 'x_center': 'x',
        'y_c': 'y', 'yc': 'y', 'y_center': 'y',
        'z_c': 'z', 'zc': 'z', 'z_center': 'z',

        # dip direction
        'azimut': 'dip_direction',
        'azimuth': 'dip_direction',
        'az': 'dip_direction',
        'dipdirection': 'dip_direction',
        'dip_dir': 'dip_direction',
        'direction': 'dip_direction',

        # dip angle
        'pendage': 'dip',
        'dip_angle': 'dip',
        'inclinaison': 'dip',

        # length
        'length': 'longueur',
        'len': 'longueur',
        'trace_length': 'longueur',
        'longueur': 'longueur',

        # aperture
        'aperture_m': 'aperture',
        'ouverture': 'aperture',
        'hydraulic_aperture': 'aperture',

        # fracture type
        'type': 'fracture_type',
        'family': 'fracture_type',
        'famille': 'fracture_type',
        'joint_set': 'fracture_type'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # check required columns
    required = ['x', 'y', 'z', 'dip_direction', 'dip', 'longueur']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # convert to float and clean
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # drop rows with NaN in required columns
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Warning: dropped {n_dropped} rows with missing values")

    # validate dip direction range (0-360)
    df['dip_direction'] = df['dip_direction'] % 360

    # validate dip range (0-90)
    df['dip'] = df['dip'].clip(0, 90)

    # add default aperture if not present
    if 'aperture' not in df.columns:
        if config is not None:
            df['aperture'] = _get_default_aperture(config, df)
        else:
            df['aperture'] = 1e-4
            print("  Warning: no aperture column, using default 0.1mm")

    # add default fracture_type if not present
    if 'fracture_type' not in df.columns:
        df['fracture_type'] = 'random_joint'
        print("  Warning: no fracture_type column, using 'random_joint'")

    # compute normal vector for each fracture
    df = _add_normal_vector(df)

    # reset index
    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df)} fractures from "
          f"{os.path.basename(csv_path)}")
    print(f"  Columns: {list(df.columns)}")

    return df


def read_dxf_fractures(dxf_path, fracture_type='fault_zone'):
    """
    Read fracture/fault geometry from a DXF file.
    Extracts LINE and POLYLINE entities as fracture traces.
    Computes dip direction and dip from 3D geometry.

    Parameters
    ----------
    dxf_path : str
        Path to the DXF file.
    fracture_type : str
        Type to assign to all fractures from this DXF.

    Returns
    -------
    pandas.DataFrame
        Fracture dataframe with columns:
        x, y, z, dip_direction, dip, longueur, fracture_type
        plus x1, y1, z1, x2, y2, z2 (endpoints)

    Example
    -------
    df = read_dxf_fractures('data/structural/faults.dxf', 'fault_zone')
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError(
            "ezdxf not installed. Run: pip install ezdxf"
        )

    if not os.path.exists(dxf_path):
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    segments = []

    # extract LINE entities
    for e in msp.query("LINE"):
        x1, y1, z1 = e.dxf.start
        x2, y2, z2 = e.dxf.end
        segments.append({
            'x1': float(x1), 'y1': float(y1), 'z1': float(z1),
            'x2': float(x2), 'y2': float(y2), 'z2': float(z2)
        })

    # extract LWPOLYLINE entities
    for e in msp.query("LWPOLYLINE"):
        pts = list(e.get_points("xy"))
        for i in range(len(pts) - 1):
            segments.append({
                'x1': float(pts[i][0]),
                'y1': float(pts[i][1]),
                'z1': 0.0,
                'x2': float(pts[i+1][0]),
                'y2': float(pts[i+1][1]),
                'z2': 0.0
            })

    # extract POLYLINE entities
    for e in msp.query("POLYLINE"):
        verts = list(e.vertices)
        for i in range(len(verts) - 1):
            segments.append({
                'x1': float(verts[i].dxf.location.x),
                'y1': float(verts[i].dxf.location.y),
                'z1': float(verts[i].dxf.location.z),
                'x2': float(verts[i+1].dxf.location.x),
                'y2': float(verts[i+1].dxf.location.y),
                'z2': float(verts[i+1].dxf.location.z)
            })

    if not segments:
        raise ValueError(
            f"No LINE or POLYLINE entities found in {dxf_path}"
        )

    df = pd.DataFrame(segments)

    # calculate center coordinates
    df['x'] = (df['x1'] + df['x2']) / 2
    df['y'] = (df['y1'] + df['y2']) / 2
    df['z'] = (df['z1'] + df['z2']) / 2

    dx = df['x2'] - df['x1']
    dy = df['y2'] - df['y1']
    dz = df['z2'] - df['z1']

    # horizontal distance
    dh = np.sqrt(dx**2 + dy**2)

    # total length
    df['longueur'] = np.sqrt(dx**2 + dy**2 + dz**2)

    # dip direction (azimuth of dip vector, perpendicular to strike)
    # strike direction first, then rotate 90 degrees for dip direction
    strike = np.degrees(np.arctan2(dx, dy)) % 360
    df['dip_direction'] = (strike + 90) % 360

    # dip angle (degrees from horizontal)
    df['dip'] = np.degrees(np.arctan2(np.abs(dz), dh))

    # fracture type
    df['fracture_type'] = fracture_type

    # add normal vector
    df = _add_normal_vector(df)

    print(f"  Loaded {len(df)} fracture segments from "
          f"{os.path.basename(dxf_path)}")

    return df


def read_piezometer_csv(csv_path):
    """
    Read piezometer pressure data from CSV file.

    Expected columns:
        piezometer_id  - unique identifier
        x, y, z        - location coordinates (m)
        pressure_pa    - measured pressure (Pa)
        date           - measurement date (optional)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Piezometer dataframe.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    column_mapping = {
        'id': 'piezometer_id',
        'piezo_id': 'piezometer_id',
        'pressure': 'pressure_pa',
        'head': 'hydraulic_head_m',
        'water_level': 'hydraulic_head_m',
        'niveau_eau': 'hydraulic_head_m'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # convert hydraulic head to pressure if needed
    if ('hydraulic_head_m' in df.columns and
            'pressure_pa' not in df.columns):
        rho = 1000.0
        g = 9.81
        df['pressure_pa'] = df['hydraulic_head_m'] * rho * g
        print("  Converted hydraulic head to pressure (Pa)")

    print(f"  Loaded {len(df)} piezometer readings from "
          f"{os.path.basename(csv_path)}")

    return df


def print_fracture_summary(df):
    """
    Print a summary of the fracture dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Fracture dataframe.
    """
    print("=" * 50)
    print("  FRACTURE DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total fractures  : {len(df)}")
    print(f"\n  Geometry:")
    print(f"    X range : {df['x'].min():.1f} to {df['x'].max():.1f} m")
    print(f"    Y range : {df['y'].min():.1f} to {df['y'].max():.1f} m")
    print(f"    Z range : {df['z'].min():.1f} to {df['z'].max():.1f} m")
    print(f"\n  Orientation:")
    print(f"    Dip direction : {df['dip_direction'].min():.1f} "
          f"to {df['dip_direction'].max():.1f} deg")
    print(f"    Dip           : {df['dip'].min():.1f} "
          f"to {df['dip'].max():.1f} deg")
    print(f"\n  Length:")
    print(f"    Min  : {df['longueur'].min():.2f} m")
    print(f"    Max  : {df['longueur'].max():.2f} m")
    print(f"    Mean : {df['longueur'].mean():.2f} m")
    if 'aperture' in df.columns:
        print(f"\n  Aperture:")
        print(f"    Min  : {df['aperture'].min()*1000:.3f} mm")
        print(f"    Max  : {df['aperture'].max()*1000:.3f} mm")
        print(f"    Mean : {df['aperture'].mean()*1000:.3f} mm")
    if 'fracture_type' in df.columns:
        print(f"\n  Fracture types:")
        for ftype, count in df['fracture_type'].value_counts().items():
            print(f"    {ftype:<25} : {count}")
    print("=" * 50)


# private helper functions

def _add_normal_vector(df):
    """
    Compute unit normal vector for each fracture from
    dip direction and dip angle.

    Normal vector convention (pointing upward):
        nx = sin(dip) * sin(dip_direction)
        ny = sin(dip) * cos(dip_direction)
        nz = cos(dip)

    Parameters
    ----------
    df : pandas.DataFrame
        Fracture dataframe with dip_direction and dip columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with added nx, ny, nz columns.
    """
    dip_rad = np.radians(df['dip'])
    dip_dir_rad = np.radians(df['dip_direction'])

    df['nx'] = np.sin(dip_rad) * np.sin(dip_dir_rad)
    df['ny'] = np.sin(dip_rad) * np.cos(dip_dir_rad)
    df['nz'] = np.cos(dip_rad)

    return df


def _get_default_aperture(config, df):
    """
    Get default aperture values from config based on fracture type.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    df : pandas.DataFrame
        Fracture dataframe with fracture_type column.

    Returns
    -------
    pandas.Series
        Aperture values for each fracture.
    """
    fracture_types = config.get('fracture_types', {})
    default = 1e-4

    if 'fracture_type' not in df.columns:
        return pd.Series([default] * len(df))

    apertures = []
    for ftype in df['fracture_type']:
        if ftype in fracture_types:
            apertures.append(
                fracture_types[ftype].get('aperture_m', default)
            )
        else:
            apertures.append(default)

    return pd.Series(apertures)