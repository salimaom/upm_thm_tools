"""
dfn_local.py
============
Local scale DFN pipe network builder.

Builds 1D pipe network for discrete fracture network
around underground excavations.

Pipe generation logic (from Exemple_pipe_channel_tunnel_v3.py):
    For each pair of intersecting fractures (i, j):
        1. Find midpoint of intersection line segment
        2. Create pipe: center_i → midpoint
        3. Create pipe: center_j → midpoint

    For each fracture intersecting tunnel/excavation:
        1. Find intersection point on tunnel wall
        2. Create pipe: center_i → tunnel_wall_point

Author: Salim Hammoum
Polytechnique Montreal - 2026

References:
    Ren et al. (2017) - Unified Pipe Network Method
    Chen et al. (2018) - T-H coupling in fractured rock
"""

import numpy as np
import pandas as pd
from itertools import combinations


# ─────────────────────────────────────────────────────────────────
# NODE CLASS
# ─────────────────────────────────────────────────────────────────

class Node:
    """
    Represents a node in the pipe network.

    Attributes
    ----------
    node_id : int
        Unique identifier.
    x, y, z : float
        3D coordinates.
    node_type : str
        Type of node:
        'fracture_center'    - center of a fracture disc
        'intersection'       - midpoint of fracture-fracture intersection
        'tunnel_wall'        - point on tunnel/excavation wall
        'matrix'             - rock matrix node (Voronoi)
    fracture_id : int or None
        Index of associated fracture (-1 for intersection nodes).
    pressure : float
        Fluid pressure (Pa). Unknown to solve.
    temperature_fluid : float
        Fluid temperature (K).
    temperature_solid : float
        Solid/rock temperature (K).
    is_boundary : bool
        True if this node has a prescribed pressure (Dirichlet BC).
    boundary_value : float
        Prescribed pressure value if is_boundary is True.
    """

    def __init__(self, node_id, x, y, z, node_type='fracture_center',
                 fracture_id=None):
        self.node_id          = node_id
        self.x                = float(x)
        self.y                = float(y)
        self.z                = float(z)
        self.node_type        = node_type
        self.fracture_id      = fracture_id
        self.pressure         = 0.0
        self.temperature_fluid = 273.15
        self.temperature_solid = 273.15
        self.is_boundary      = False
        self.boundary_value   = 0.0

    @property
    def coords(self):
        """Return coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return (f"Node(id={self.node_id}, "
                f"type={self.node_type}, "
                f"xyz=({self.x:.1f},{self.y:.1f},{self.z:.1f}))")


# ─────────────────────────────────────────────────────────────────
# PIPE CLASS
# ─────────────────────────────────────────────────────────────────

class Pipe:
    """
    Represents a pipe segment in the network.

    Attributes
    ----------
    pipe_id : int
        Unique identifier.
    node_i, node_j : int
        Node IDs at each end of the pipe.
    length : float
        Pipe length (m).
    aperture : float
        Hydraulic aperture (m).
    width : float
        Equivalent pipe width (m) for cubic law.
    pipe_type : str
        Type of pipe:
        'fracture'   - within fracture network
        'tunnel'     - fracture center to tunnel wall
        'matrix'     - rock matrix heat pipe
    fracture_i : int
        Fracture index associated with node_i.
    fracture_j : int
        Fracture index associated with node_j.
    conductance : float
        Hydraulic conductance K = a³·w/(12·mu·L)
        Computed after aperture and geometry are set.
    """

    def __init__(self, pipe_id, node_i, node_j, length,
                 aperture, width, pipe_type='fracture',
                 fracture_i=-1, fracture_j=-1):
        self.pipe_id    = pipe_id
        self.node_i     = node_i
        self.node_j     = node_j
        self.length     = float(length)
        self.aperture   = float(aperture)
        self.width      = float(width)
        self.pipe_type  = pipe_type
        self.fracture_i = fracture_i
        self.fracture_j = fracture_j
        self.conductance = 0.0

    def __repr__(self):
        return (f"Pipe(id={self.pipe_id}, "
                f"nodes=({self.node_i},{self.node_j}), "
                f"L={self.length:.2f}m, "
                f"a={self.aperture*1000:.3f}mm)")


# ─────────────────────────────────────────────────────────────────
# FRACTURE GEOMETRY
# ─────────────────────────────────────────────────────────────────

def compute_normal_vector(dip_direction_deg, dip_deg):
    """
    Compute unit normal vector from dip direction and dip angle.

    Geological convention:
        dip_direction : azimuth of dip vector (0=North, 90=East)
        dip           : angle from horizontal (0=flat, 90=vertical)

    Normal vector points upward out of fracture plane:
        nx = sin(dip) * sin(dip_direction)
        ny = sin(dip) * cos(dip_direction)
        nz = cos(dip)

    Parameters
    ----------
    dip_direction_deg : float
        Dip direction in degrees.
    dip_deg : float
        Dip angle in degrees.

    Returns
    -------
    numpy.ndarray
        Unit normal vector (3,).
    """
    d  = np.radians(dip_deg)
    dd = np.radians(dip_direction_deg)
    n  = np.array([
        np.sin(d) * np.sin(dd),
        np.sin(d) * np.cos(dd),
        np.cos(d)
    ])
    return n / np.linalg.norm(n)


def compute_fracture_radius(longueur):
    """
    Compute fracture disc radius from trace length.
    Assumes circular disc: radius = length / 2.

    Parameters
    ----------
    longueur : float
        Fracture trace length (m).

    Returns
    -------
    float
        Fracture radius (m).
    """
    return longueur / 2.0


# ─────────────────────────────────────────────────────────────────
# INTERSECTION DETECTION
# ─────────────────────────────────────────────────────────────────

def find_intersection_midpoint(center_a, normal_a, radius_a,
                                center_b, normal_b, radius_b):
    """
    Find midpoint of intersection line segment between two fracture discs.

    Algorithm:
        1. Quick distance filter
        2. Find intersection line direction (cross product of normals)
        3. Find point on intersection line (least squares)
        4. Find overlap interval on the line
        5. Compute midpoint of overlap
        6. Validate midpoint is inside both discs

    Parameters
    ----------
    center_a, center_b : numpy.ndarray
        Centers of fracture discs (3,).
    normal_a, normal_b : numpy.ndarray
        Unit normal vectors of fracture planes (3,).
    radius_a, radius_b : float
        Radii of fracture discs (m).

    Returns
    -------
    numpy.ndarray or None
        Midpoint of intersection segment (3,),
        or None if no valid intersection.
    """
    # quick distance filter
    dist = np.linalg.norm(center_a - center_b)
    if dist > radius_a + radius_b + 1e-6:
        return None

    # intersection line direction
    d_vec = np.cross(normal_a, normal_b)
    norm_d = np.linalg.norm(d_vec)
    if norm_d < 1e-10:
        return None
    d_vec /= norm_d

    # point on intersection line (least squares)
    A_ls = np.array([normal_a, normal_b])
    b_ls = np.array([normal_a @ center_a, normal_b @ center_b])
    X0   = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0]

    # project midpoint of centers onto line
    mid_c = (center_a + center_b) / 2
    X0   += ((mid_c - X0) @ d_vec) * d_vec

    # overlap interval
    ta = (center_a - X0) @ d_vec
    tb = (center_b - X0) @ d_vec
    t0 = max(ta - radius_a, tb - radius_b)
    t1 = min(ta + radius_a, tb + radius_b)

    if t0 >= t1 - 1e-9:
        return None

    midpoint = X0 + ((t0 + t1) / 2) * d_vec

    # validate midpoint is inside both discs
    def in_disc(pt, center, normal, radius):
        proj = pt - np.dot(pt - center, normal) * normal
        return np.linalg.norm(proj - center) <= radius * 1.05

    if not in_disc(midpoint, center_a, normal_a, radius_a):
        return None
    if not in_disc(midpoint, center_b, normal_b, radius_b):
        return None

    return midpoint


# ─────────────────────────────────────────────────────────────────
# PIPE NETWORK BUILDER
# ─────────────────────────────────────────────────────────────────

def build_local_pipe_network(df, config):
    """
    Build 1D local DFN pipe network from fracture dataframe.

    For each pair of intersecting fractures (i, j):
        Creates two pipes:
            center_i → intersection_midpoint
            center_j → intersection_midpoint

    Parameters
    ----------
    df : pandas.DataFrame
        Fracture dataframe from readers.read_fracture_csv().
        Required columns: x, y, z, dip_direction, dip,
                          longueur, aperture, fracture_type
    config : dict
        Configuration dictionary from config.load_config().

    Returns
    -------
    nodes : list of Node
        All nodes in the pipe network.
    pipes : list of Pipe
        All pipe segments in the network.

    Example
    -------
    nodes, pipes = build_local_pipe_network(df, config)
    print(f"Nodes: {len(nodes)}, Pipes: {len(pipes)}")
    """

    nodes    = []
    pipes    = []
    node_id  = 0
    pipe_id  = 0

    # get fluid viscosity from config
    fluid = config.get('fluid', {}).get('water', {})
    mu    = fluid.get('viscosity', 1e-3)

    # ── Step 1: Create fracture center nodes ─────────────────────
    centers = []
    normals = []
    radii   = []
    center_node_ids = []

    for i, row in df.iterrows():
        # compute geometry
        normal = compute_normal_vector(
            row['dip_direction'], row['dip']
        )
        radius = compute_fracture_radius(row['longueur'])

        centers.append(np.array([row['x'], row['y'], row['z']]))
        normals.append(normal)
        radii.append(radius)

        # create center node
        node = Node(
            node_id = node_id,
            x       = row['x'],
            y       = row['y'],
            z       = row['z'],
            node_type   = 'fracture_center',
            fracture_id = i
        )
        nodes.append(node)
        center_node_ids.append(node_id)
        node_id += 1

    centers = np.array(centers)
    normals = np.array(normals)
    radii   = np.array(radii)

    n_fractures = len(df)
    apertures   = df['aperture'].values

    print(f"  Created {n_fractures} fracture center nodes")

    # ── Step 2: Find intersections and create pipes ───────────────
    n_intersections = 0
    n_pipes         = 0

    for i, j in combinations(range(n_fractures), 2):

        midpoint = find_intersection_midpoint(
            centers[i], normals[i], radii[i],
            centers[j], normals[j], radii[j]
        )

        if midpoint is None:
            continue

        # create intersection node
        inter_node = Node(
            node_id     = node_id,
            x           = midpoint[0],
            y           = midpoint[1],
            z           = midpoint[2],
            node_type   = 'intersection',
            fracture_id = -1
        )
        nodes.append(inter_node)
        inter_node_id = node_id
        node_id += 1
        n_intersections += 1

        # average aperture at intersection
        aperture_i = apertures[i]
        aperture_j = apertures[j]

        # pipe 1: center_i → midpoint
        length_i = np.linalg.norm(centers[i] - midpoint)
        if length_i < 1e-6:
            length_i = 1e-6  # minimum pipe length

        pipe_i = Pipe(
            pipe_id   = pipe_id,
            node_i    = center_node_ids[i],
            node_j    = inter_node_id,
            length    = length_i,
            aperture  = aperture_i,
            width     = aperture_i,
            pipe_type = 'fracture',
            fracture_i = i,
            fracture_j = j
        )
        pipes.append(pipe_i)
        pipe_id += 1
        n_pipes += 1

        # pipe 2: center_j → midpoint
        length_j = np.linalg.norm(centers[j] - midpoint)
        if length_j < 1e-6:
            length_j = 1e-6  # minimum pipe length

        pipe_j = Pipe(
            pipe_id   = pipe_id,
            node_i    = center_node_ids[j],
            node_j    = inter_node_id,
            length    = length_j,
            aperture  = aperture_j,
            width     = aperture_j,
            pipe_type = 'fracture',
            fracture_i = j,
            fracture_j = i
        )
        pipes.append(pipe_j)
        pipe_id += 1
        n_pipes += 1

    print(f"  Found {n_intersections} fracture intersections")
    print(f"  Created {n_pipes} fracture pipes")

    return nodes, pipes


def print_network_summary(nodes, pipes):
    """
    Print a summary of the pipe network.

    Parameters
    ----------
    nodes : list of Node
        All nodes in the network.
    pipes : list of Pipe
        All pipes in the network.
    """
    print("=" * 50)
    print("  PIPE NETWORK SUMMARY")
    print("=" * 50)

    # count node types
    type_counts = {}
    for node in nodes:
        t = node.node_type
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  NODES: {len(nodes)} total")
    for ntype, count in type_counts.items():
        print(f"    {ntype:<25} : {count}")

    # count pipe types
    pipe_counts = {}
    for pipe in pipes:
        t = pipe.pipe_type
        pipe_counts[t] = pipe_counts.get(t, 0) + 1

    print(f"\n  PIPES: {len(pipes)} total")
    for ptype, count in pipe_counts.items():
        print(f"    {ptype:<25} : {count}")

    if pipes:
        lengths  = [p.length for p in pipes]
        apertures = [p.aperture for p in pipes]
        print(f"\n  PIPE LENGTHS:")
        print(f"    Min  : {min(lengths):.2f} m")
        print(f"    Max  : {max(lengths):.2f} m")
        print(f"    Mean : {sum(lengths)/len(lengths):.2f} m")
        print(f"\n  PIPE APERTURES:")
        print(f"    Min  : {min(apertures)*1000:.3f} mm")
        print(f"    Max  : {max(apertures)*1000:.3f} mm")
        print(f"    Mean : {sum(apertures)/len(apertures)*1000:.3f} mm")

    print("=" * 50)


def get_nodes_dataframe(nodes):
    """
    Convert nodes list to pandas DataFrame for easy inspection.

    Parameters
    ----------
    nodes : list of Node

    Returns
    -------
    pandas.DataFrame
    """
    data = []
    for n in nodes:
        data.append({
            'node_id'     : n.node_id,
            'x'           : n.x,
            'y'           : n.y,
            'z'           : n.z,
            'node_type'   : n.node_type,
            'fracture_id' : n.fracture_id,
            'is_boundary' : n.is_boundary
        })
    return pd.DataFrame(data)


def get_pipes_dataframe(pipes):
    """
    Convert pipes list to pandas DataFrame for easy inspection.

    Parameters
    ----------
    pipes : list of Pipe

    Returns
    -------
    pandas.DataFrame
    """
    data = []
    for p in pipes:
        data.append({
            'pipe_id'    : p.pipe_id,
            'node_i'     : p.node_i,
            'node_j'     : p.node_j,
            'length_m'   : p.length,
            'aperture_mm': p.aperture * 1000,
            'pipe_type'  : p.pipe_type,
            'fracture_i' : p.fracture_i,
            'fracture_j' : p.fracture_j
        })
    return pd.DataFrame(data)