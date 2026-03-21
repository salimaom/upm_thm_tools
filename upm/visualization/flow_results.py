"""
flow_results.py
===============
Visualization of UPM flow solver results.

Author: Salim Hammoum
Polytechnique Montreal - 2026
"""

import numpy as np
import plotly.graph_objects as go


def plot_pipe_network_3d(nodes, pipes, Phi, Q,
                          title="UPM Flow Results",
                          save_html=None):
    """
    Create interactive 3D visualization of pipe network
    with pressure field and flow rates.

    Nodes colored by pressure (blue=low, red=high)
    Pipes shown as grey lines
    Boundary nodes shown as yellow crosses

    Parameters
    ----------
    nodes : list of Node
    pipes : list of Pipe
    Phi : numpy.ndarray
        Nodal pressure solution (Pa).
    Q : numpy.ndarray
        Pipe flow rates (m3/s).
    title : str
    save_html : str or None
        Path to save HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    traces = []
    p_min  = Phi.min()
    p_max  = Phi.max()
    q_max  = np.abs(Q).max() if len(Q) > 0 else 1.0

    # pipe traces
    xs_pipes = []
    ys_pipes = []
    zs_pipes = []

    for idx, pipe in enumerate(pipes):
        ni = nodes[pipe.node_i]
        nj = nodes[pipe.node_j]
        xs_pipes += [ni.x, nj.x, None]
        ys_pipes += [ni.y, nj.y, None]
        zs_pipes += [ni.z, nj.z, None]

    traces.append(go.Scatter3d(
        x=xs_pipes, y=ys_pipes, z=zs_pipes,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.6)', width=3),
        name='Pipes',
        hoverinfo='skip'
    ))

    # fracture center nodes
    fc_nodes = [n for n in nodes
                if n.node_type == 'fracture_center']
    if fc_nodes:
        traces.append(go.Scatter3d(
            x=[n.x for n in fc_nodes],
            y=[n.y for n in fc_nodes],
            z=[n.z for n in fc_nodes],
            mode='markers',
            marker=dict(
                size=10,
                color=[n.pressure for n in fc_nodes],
                colorscale='RdBu_r',
                cmin=p_min,
                cmax=p_max,
                showscale=True,
                colorbar=dict(title="Pressure (Pa)", x=1.02),
                line=dict(color='white', width=1)
            ),
            text=[
                f"Node {n.node_id}<br>"
                f"Fracture center<br>"
                f"P = {n.pressure:.3e} Pa"
                for n in fc_nodes
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Fracture centers'
        ))

    # intersection nodes
    int_nodes = [n for n in nodes
                 if n.node_type == 'intersection']
    if int_nodes:
        traces.append(go.Scatter3d(
            x=[n.x for n in int_nodes],
            y=[n.y for n in int_nodes],
            z=[n.z for n in int_nodes],
            mode='markers',
            marker=dict(
                size=6,
                color=[n.pressure for n in int_nodes],
                colorscale='RdBu_r',
                cmin=p_min,
                cmax=p_max,
                showscale=False,
                symbol='diamond',
                line=dict(color='white', width=0.5)
            ),
            text=[
                f"Node {n.node_id}<br>"
                f"Intersection<br>"
                f"P = {n.pressure:.3e} Pa"
                for n in int_nodes
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Intersections'
        ))

    # boundary nodes
    bc_nodes = [n for n in nodes if n.is_boundary]
    if bc_nodes:
        traces.append(go.Scatter3d(
            x=[n.x for n in bc_nodes],
            y=[n.y for n in bc_nodes],
            z=[n.z for n in bc_nodes],
            mode='markers',
            marker=dict(
                size=14,
                color='yellow',
                symbol='cross',
                line=dict(color='black', width=2)
            ),
            text=[
                f"Node {n.node_id}<br>"
                f"BOUNDARY<br>"
                f"P = {n.boundary_value:.3e} Pa"
                for n in bc_nodes
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Boundary nodes'
        ))

    # layout
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=14)),
            scene=dict(
                xaxis=dict(title='X (m)'),
                yaxis=dict(title='Y (m)'),
                zaxis=dict(title='Z (m)'),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            legend=dict(
                x=0, y=1,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            width=900,
            height=700
        )
    )

    if save_html:
        fig.write_html(save_html)
        print(f"  Saved to: {save_html}")

    fig.show()
    return fig


def plot_pressure_profile(nodes, Phi, axis='x',
                           title="Pressure Profile"):
    """
    Plot pressure vs position along one axis.

    Parameters
    ----------
    nodes : list of Node
    Phi : numpy.ndarray
    axis : str
        'x', 'y', or 'z'
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    coords = {
        'x': [n.x for n in nodes],
        'y': [n.y for n in nodes],
        'z': [n.z for n in nodes]
    }
    pos = coords[axis]

    fc_pos = [pos[i] for i, n in enumerate(nodes)
              if n.node_type == 'fracture_center']
    fc_p   = [Phi[i] for i, n in enumerate(nodes)
              if n.node_type == 'fracture_center']

    int_pos = [pos[i] for i, n in enumerate(nodes)
               if n.node_type == 'intersection']
    int_p   = [Phi[i] for i, n in enumerate(nodes)
               if n.node_type == 'intersection']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=int_pos, y=int_p,
        mode='markers',
        marker=dict(size=6, color='blue', symbol='diamond'),
        name='Intersection nodes'
    ))

    fig.add_trace(go.Scatter(
        x=fc_pos, y=fc_p,
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Fracture centers'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f'{axis.upper()} coordinate (m)',
        yaxis_title='Pressure (Pa)',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.95)
    )

    fig.show()
    return fig


def plot_flow_rates(pipes, Q, title="Pipe Flow Rates"):
    """
    Bar chart of absolute flow rates in all pipes.

    Parameters
    ----------
    pipes : list of Pipe
    Q : numpy.ndarray
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    pipe_ids = [p.pipe_id for p in pipes]
    q_abs    = np.abs(Q)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pipe_ids,
        y=q_abs,
        marker_color='steelblue',
        text=[f"{q:.2e}" for q in q_abs],
        textposition='outside',
        name='|Flow rate|'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Pipe ID',
        yaxis_title='|Flow rate| (m3/s)',
        yaxis_type='log',
        width=900,
        height=500
    )

    fig.show()
    return fig