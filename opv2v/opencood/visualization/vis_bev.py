# vis_bev.py
# Visualize BEV: each ego vehicle + the vehicles it sees (in distinct colors)

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import cos, sin

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.camera_only.base_camera_dataset import BaseCameraDataset
from opencood.utils.box_utils import boxes2d_to_corners2d


def vis_parser():
    parser = argparse.ArgumentParser(description="BEV visualization per ego")
    parser.add_argument('--scene', type=int, default=4,
                        help='The ith scene to visualize')
    parser.add_argument('--sample', type=int, default=10,
                        help='The jth sample in the scene')
    return parser.parse_args()


def extract_xy_yaw_from_pose(pose):
    """
    Extract (x, y, yaw) from a 6-DoF pose: [x, y, z, roll, pitch, yaw]
    """
    p = np.array(pose)
    if p.ndim == 1 and p.size >= 6:
        return p[0], p[1], p[5]
    raise ValueError(f"Unexpected lidar_pose format: {pose}")


def transform_boxes_to_ego_frame(boxes_cav, cav_pose, ego_pose):
    """
    Transform boxes from one CAV's local frame into the main ego BEV frame.
    boxes_cav: (N,7) [x,y,z,dx,dy,dz,yaw] in cav frame
    cav_pose/ego_pose: [x,y,z,roll,pitch,yaw] in world
    Returns: (N,5) [x, y, dx, dy, yaw] in ego BEV frame
    """
    cav_x, cav_y, cav_yaw = extract_xy_yaw_from_pose(cav_pose)
    ego_x, ego_y, ego_yaw = extract_xy_yaw_from_pose(ego_pose)

    centers = boxes_cav[:, :2]
    dims = boxes_cav[:, 3:5]
    yaws = boxes_cav[:, 6]

    out = []
    for (cx, cy), (dx, dy), oy in zip(centers, dims, yaws):
        # CAV local → world
        wx = cav_x + cos(cav_yaw)*cx - sin(cav_yaw)*cy
        wy = cav_y + sin(cav_yaw)*cx + cos(cav_yaw)*cy
        wyaw = cav_yaw + oy
        # world → ego local
        dxw = wx - ego_x
        dyw = wy - ego_y
        ex = dxw*cos(-ego_yaw) - dyw*sin(-ego_yaw)
        ey = dxw*sin(-ego_yaw) + dyw*cos(-ego_yaw)
        eyaw = wyaw - ego_yaw
        out.append([ex, ey, dx, dy, eyaw])
    return np.array(out)


def draw_bbox(ax, boxes2d, color='red'):
    """
    Draw 2D BEV boxes: boxes2d = (N,5) [x,y,dx,dy,yaw].
    """
    if boxes2d is None or len(boxes2d) == 0:
        return
    corners = boxes2d_to_corners2d(boxes2d, order='lwh')  # (N,4,2)
    for c in corners:
        poly = patches.Polygon(c, closed=True,
                               edgecolor=color, facecolor='none', linewidth=1)
        ax.add_patch(poly)


if __name__ == '__main__':
    opt = vis_parser()
    base = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(base, '../hypes_yaml/opcamera/base_camera.yaml'))

    dataset = BaseCameraDataset(params, train=True, visualize=True)
    data = dataset.get_sample(opt.scene, opt.sample)

    # Identify main ego vehicle
    ego_pose = None
    for cav_id, info in data.items():
        if info.get('ego', False):
            ego_pose = info['params']['lidar_pose']
            break
    if ego_pose is None:
        raise RuntimeError("No ego vehicle in this frame.")

    # Plot setup
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"Scene {opt.scene}, Sample {opt.sample}")
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (right)")
    ax.set_aspect('equal')
    ax.grid(True)

    # Color palette for each ego
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for idx, (cav_id, info) in enumerate(data.items()):
        cav_pose = info['params']['lidar_pose']
        x, y, yaw = extract_xy_yaw_from_pose(cav_pose)
        # Relative to main ego
        ego_x, ego_y, _ = extract_xy_yaw_from_pose(ego_pose)
        rel_x, rel_y = x - ego_x, y - ego_y

        c = colors[idx % len(colors)]
        # Draw ego vehicle
        rect = patches.Rectangle((rel_x-1.5, rel_y-1.0), 3.0, 2.0,
                                 angle=np.degrees(yaw),
                                 linewidth=1, edgecolor='black',
                                 facecolor=c, alpha=0.4)
        ax.add_patch(rect)
        ax.text(rel_x, rel_y, f"EGO{cav_id}", color='white',
                 fontsize=8, ha='center', va='center')

        # Draw that ego's observed vehicles
        obs = info.get('object_bbx_cav', None)
        if obs is not None and len(obs):
            boxes2d = transform_boxes_to_ego_frame(
                np.array(obs), cav_pose, ego_pose)
            draw_bbox(ax, boxes2d, color=c)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.tight_layout()
    out_file = f'scene{opt.scene}_sample{opt.sample}_bev.png'
    plt.savefig(out_file)
    print(f"[INFO] Saved BEV visualization: {out_file}")