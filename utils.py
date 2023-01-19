import gtsam
import numpy as np
import plotly.graph_objects as go
import math

def construct_local_shapes(env, image_colors):
    local_shapes = []
    # draw walls
    for wall in env.wall_list:
        local_shapes.append(wall.get_rect())

    # draw images
    for image in env.image_list:
        image_color = list(image_colors[image.image_label])
        image_color.append(0.5)
        local_shapes.append(image.get_rect(image_color))
    return local_shapes

def generate_play_button(num_steps, duration):
    # Show play button when enough frames
    play_button = []
    if num_steps >= 2:
        play_button = [dict(
            buttons=[dict(
                args=[None, dict(
                    frame=dict(duration=1000 * duration),
                    fromcurrent=True
                )],
                label="Play",
                method="animate"
            )],
            direction="left",
            pad=dict(r=10, t=30),
            showactive=False,
            type="buttons",
            x=0.05,
            xanchor="right",
            y=0,
            yanchor="top",
            font=dict(
                family="Courier New, monospace",
                color='rgb(230, 230, 230)'
            )
        )]
    return play_button

def generate_layout(env, local_shapes, play_button, labels, duration, show_grid_lines):
    layout = go.Layout(
        xaxis=dict(dtick=1.0,
                range=[-1, env.room_size[0]+ 1],
                showticklabels=show_grid_lines,
                showgrid=show_grid_lines,
                gridcolor='rgba(175, 175, 175, 255)',
                zeroline=False),
        yaxis=dict(dtick=1.0,
                range=[-1, env.room_size[1] + 1],
                showticklabels=show_grid_lines,
                showgrid=show_grid_lines,
                gridcolor='rgba(175, 175, 175, 255)',
                zeroline=False,
                scaleanchor="x",
                scaleratio=1),
        margin=dict(r=30, l=30, b=30, t=30),
        paper_bgcolor='rgba(50, 50, 60, 255)',
        plot_bgcolor='rgba(50, 50, 60, 255)',
        font=dict(
            family="Courier New, monospace",
            color='rgba(230, 230, 230, 255)' if show_grid_lines else 'rgba(255, 255, 255, 0)'
        ),
        shapes=local_shapes,
        dragmode='pan',
        hovermode=False,
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(
                    family="Courier New, monospace",
                    color='rgb(230, 230, 230)'
                ),
                prefix="Step ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=0),
            len=0.95,
            x=0.05,
            y=0,
            font=dict(
                family="Courier New, monospace",
                color='rgb(230, 230, 230)'
            ),
            steps=[dict(
                    args=[[local_label], dict(
                        frame=dict(duration=1000 * duration),
                        mode="immediate"
                    )],
                    label=local_label,
                    method="animate") for local_label in labels]
        )],
        updatemenus=play_button,
    )
    return layout


def visualize(env, gt_poses = [], show_grid_lines=True, duration=0.3):
    image_colors = [[255, 0, 0], [0, 255, 0], [210, 105, 30],
                    [255, 255, 0], [0, 255, 255], [255, 0, 255],
                    [255, 165, 0], [255, 0, 127], [0, 127, 255],
                    [0, 255, 127]]

    num_steps = len(gt_poses)
    labels = [str(i) for i in range(num_steps)]
    first_data = None
    frames = []
    for frame_idx in range(num_steps):
        xs_gt = []
        ys_gt = []
        for pose_idx in range(frame_idx+1):
            xs_gt.append(gt_poses[pose_idx].x())
            ys_gt.append(gt_poses[pose_idx].y())

        pose = gt_poses[frame_idx]
        v1 = pose.transformFrom(np.array([-0.3, -0.2]))
        v2 = pose.transformFrom(np.array([-0.3, 0.2]))
        v3 = pose.transformFrom(np.array([0.3, 0.2]))
        v4 = pose.transformFrom(np.array([0.3, -0.2]))
        xs_car = [v1[0], v2[0], v3[0], v4[0]]
        ys_car = [v1[1], v2[1], v3[1], v4[1]]

        data = [
            go.Scatter(
                x=xs_gt, y=ys_gt,
                mode='lines+markers',
                marker=dict(color='Blue', size=10),
                showlegend=False
            ),
            go.Scatter(
                x=xs_car, y=ys_car,
                mode="markers",
                marker=dict(color='Red', size=4),
                fill="toself",
                showlegend=False
            ),
        ]

        frames.append(go.Frame(name=labels[frame_idx], data=data))
        if first_data is None:
            first_data = data

    play_button = generate_play_button(num_steps, duration)
    local_shapes = construct_local_shapes(env, image_colors)
    layout = generate_layout(env, local_shapes, play_button, labels, duration, show_grid_lines)

    fig = go.Figure(data=first_data, layout=layout, frames=frames)
    fig.show()

def visualize_tree(env, tree = [], values_list=None, show_grid_lines=True, duration=0):
    image_colors = [[255, 0, 0], [0, 255, 0], [210, 105, 30],
                    [255, 255, 0], [0, 255, 255], [255, 0, 255],
                    [255, 165, 0], [255, 0, 127], [0, 127, 255],
                    [0, 255, 127]]

    num_steps = len(tree)
    labels = [str(i) for i in range(num_steps)]
    first_data = None
    frames = []
    data = []
    for i, point in enumerate(tree):
      curr = point
      parent = point.parent
 
      if parent:
        xs = [curr.x, parent.x]
        ys = [curr.y, parent.y]
      else:
        xs = [curr.x]
        ys = [curr.y]
      data.append(go.Scatter(
                  x=xs, y=ys,
                  mode='lines+markers',
                  marker=dict(color='Blue', size=10),
                  showlegend=False
              ))
      frames.append(go.Frame(name=labels[i], data=data))
      if first_data is None:
            first_data = data

    play_button = generate_play_button(num_steps, duration)
    local_shapes = construct_local_shapes(env, image_colors)
    layout = generate_layout(env, local_shapes, play_button, labels, duration, show_grid_lines)

    fig = go.Figure(data=first_data, layout=layout)
    fig.show()

def visualize_path(env, path = [], show_grid_lines=True, duration=0.3):
    image_colors = [[255, 0, 0], [0, 255, 0], [210, 105, 30],
                    [255, 255, 0], [0, 255, 255], [255, 0, 255],
                    [255, 165, 0], [255, 0, 127], [0, 127, 255],
                    [0, 255, 127]]

    num_steps = len(path)
    labels = [str(i) for i in range(num_steps)]
    first_data = None
    frames = []
    for frame_idx in range(num_steps):
        xs_gt = []
        ys_gt = []
        for pose_idx in range(frame_idx+1):
            xs_gt.append(path[pose_idx].x)
            ys_gt.append(path[pose_idx].y)

        data = [
            go.Scatter(
                x=xs_gt, y=ys_gt,
                mode='lines+markers',
                marker=dict(color='Blue', size=10),
                showlegend=False
            ),
        ]

        frames.append(go.Frame(name=labels[frame_idx], data=data))
        if first_data is None:
            first_data = data

    play_button = generate_play_button(num_steps, duration)
    local_shapes = construct_local_shapes(env, image_colors)
    layout = generate_layout(env, local_shapes, play_button, labels, duration, show_grid_lines)

    fig = go.Figure(data=first_data, layout=layout, frames=frames)
    fig.show()
    
    
import shapely
from shapely.geometry import LineString, Point

def line_intersection(start, end, obs_start, obs_end):
    '''Checks if given two lines intersect.

    Parameters
    ----------
    start (array): [x,y] coordinate of the start of the line
    end (array): [x,y] coordinate of the end of the line
    obs_start (array): [x,y] coordinate of the start of the obstacle line
    obs_end (array): [x,y] coordinate of the end of the obstacle line

    Return
    ------
    (boolean) True if the two lines intersect
    '''
    line1 = LineString([start, end])
    line2 = LineString([obs_start, obs_end])
    if line1.intersection(line2):
        return True
    return False  
      
