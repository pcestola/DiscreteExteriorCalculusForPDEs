import time
import numpy as np
import meshplot as mp
import ipywidgets as iw
from PIL import Image

def visualize_mesh(mesh, faces=False, boundary=False):
    p = mp.plot(mesh.vertices, mesh.faces,
                shading={
                    "colormap": "viridis", 
                    "flat": True,
                    "wireframe": faces,
                    "width": 450,
                    "height": 450,
                    "scale": 2.0,
                    "background": "#f0f0f0"
                },
                return_plot=True)
    if boundary:
        p.add_points(mesh.vertices[mesh.get_boundary()], shading=dict(point_size=0.1))

def scalar_field(mesh, field):
    p = mp.plot(mesh.vertices, mesh.faces,
                c=field,
                shading={
                    "colormap": "viridis", 
                    "flat": True,
                    "width": 450,
                    "height": 450,
                    "scale": 2.0,
                    "background": "#f0f0f0"
                },
                return_plot=True)

def vector_field(mesh, field, scale=1.0):
    p = mp.plot(mesh.vertices, mesh.faces,
                shading={
                    "colormap": "viridis", 
                    "flat": True,
                    "width": 450,
                    "height": 450,
                    "scale": 2.0,
                    "background": "#f0f0f0"
                },
                return_plot=True)
    p.add_lines(mesh.vertices, mesh.vertices + scale * field, shading={"line_color": "red"})


def animate_solution(solution, mesh, pause_time=100, cmap='viridis'):
    # Calculate the minimum and maximum values for the colorbar
    min_val = np.min(solution)
    max_val = np.max(solution)

    # Create the initial plot
    p = mp.plot(mesh.vertices, mesh.faces, 
                c=solution[0], 
                shading={
                    "colormap": cmap, 
                    "normalize": [min_val, max_val],
                    "flat": True,
                    "wireframe": False,
                    "width": 450,
                    "height": 450,
                    "scale": 2.0,
                    "background": "#f0f0f0"
                },
                return_plot=True)

    def update(step):
        p.update_object(colors=solution[step])

    play = iw.Play(
        value=0,
        min=0,
        max=solution.shape[0]-1,
        step=1,
        description="Press play",
        disabled=False,
        interval=pause_time
    )
    slider = iw.IntSlider(min=0, max=solution.shape[0]-1, value=0)
    iw.jslink((play, 'value'), (slider, 'value'))
    slider.observe(lambda change: update(change['new']), names='value')

    return iw.HBox([play, slider])


def animate_solution_normal(solution, mesh, scale=0.1, pause_time=100):
    # Ensure vertex normals and initial positions are correctly obtained
    normals = scale * mesh.get_vertex_normals(normalize=True)
    position = mesh.vertices.copy()
    
    Min = solution.min()
    Max = solution.max()
    colors = (solution-Min)/(Max-Min)

    # Create the initial plot
    p = mp.plot(position, mesh.faces, c=colors[0], shading={
        "colormap": "viridis",
        "normalize": [Min, Max],
        "cmin": Min,
        "cmax": Max,
        "flat": True,
        "wireframe": False,
        "width": 450,
        "height": 450,
        "scale": 2.0,
        "background": "#f0f0f0"}, return_plot=True)
    
    def update(step):
        updated_vertices = position + solution[step][:, np.newaxis] * normals
        p.update_object(vertices=updated_vertices, colors=colors[step])

    play = iw.Play(
        value=0,
        min=0,
        max=solution.shape[0]-1,
        step=1,
        description="Press play",
        disabled=False,
        interval=pause_time
    )
    slider = iw.IntSlider(min=0, max=solution.shape[0]-1, value=0)
    iw.jslink((play, 'value'), (slider, 'value'))
    slider.observe(lambda change: update(change['new']), names='value')

    return iw.HBox([play, slider])