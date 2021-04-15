import numpy as np
import open3d as o3d
import sys


ITOP_lines = [
        [0, 1],
        [1, 8],
        [8, 10],
        [8, 9],
        [10, 12],
        [9, 11],
        [12, 14],
        [11, 13],
        [1, 3],
        [3, 5],
        [5, 7],
        [1, 2],
        [2, 4],
        [4, 6]
    ]

PANOPTIC_lines = [
        [5, 4],
        [4, 3],
        [3, 0],
        [0, 9],
        [9, 10],
        [10, 11],
        [0, 2],
        [2, 6],
        [6, 7],
        #[7, 8], # knee - ankle
        [2, 12],
        [12, 13],
        #[13, 14], # knee - ankle
    ]



result_points = np.genfromtxt(sys.argv[1])

if result_points.shape[1] == 45:
    result_points = result_points.reshape((-1, 15, 3))
    lines = ITOP_lines
elif result_points.shape[1] == 57:
    result_points = result_points.reshape((-1, 19, 3))
    lines = PANOPTIC_lines

frames_number = result_points.shape[0]

frame_counter = 0

points = result_points[frame_counter]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

camera_parameters = "camera.json"

def draw_joints(pcd):
    def visualize(vis, frame):
        vis.clear_geometries()
        points = result_points[frame]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

        param = o3d.io.read_pinhole_camera_parameters("./camera.json")
        vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    def prev_joint(vis):
        global frame_counter
        frame_counter -= 1
        visualize(vis, frame_counter % frames_number)
        return False

    def next_joint(vis):
        global frame_counter
        frame_counter += 1
        visualize(vis, frame_counter % frames_number)
        return False

    def save_view_point(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("./camera.json", param)
        return False

    def load_view_point(vis):
        param = o3d.io.read_pinhole_camera_parameters("./camera.json")
        vis.get_view_control().convert_from_pinhole_camera_parameters(param)

    #load_view_point(pcd, camera_parameters)
    key_to_callback = {}
    key_to_callback[ord("A")] = prev_joint
    key_to_callback[ord("D")] = next_joint
    key_to_callback[ord("S")] = save_view_point
    key_to_callback[ord("L")] = load_view_point

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

draw_joints(line_set)
