import glob, os
import gzip
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from absl import flags
from absl import app
FLAGS = flags.FLAGS
flags.DEFINE_string('joints_dir', "/home/Datasets/PANOPTIC/raw/01/joints", 'Joints directory.')
flags.DEFINE_string('clouds_dir', "/home/Datasets/PANOPTIC/raw/01/point_cloud_w_header", 'Point clouds directory.')
flags.DEFINE_string('out_clouds_top', "/home/Datasets/PANOPTIC/raw/01/dataset_new/cloouds_top", 'Cut point clouds directory.')
flags.DEFINE_string('out_render_dir', "/home/Datasets/PANOPTIC/raw/01/dataset_new/render", 'Output renders directory.')
flags.DEFINE_string('out_depth_dir', "/home/Datasets/PANOPTIC/raw/01/dataset_new/depth", 'Output depth maps directory.')
flags.DEFINE_string('out_3D_dir', "/home/Datasets/PANOPTIC/raw/01/dataset_new/3D", 'Output 3D maps directory.')
flags.DEFINE_string('out_2D_dir', "/home/Datasets/PANOPTIC/raw/01/dataset_new/2D", 'Output 2D maps directory.')
flags.DEFINE_string('params_dir', "/home/Datasets/PANOPTIC/raw/01/dataset_new/cam_params/512x512", 'Camera parameters directory.')
flags.DEFINE_integer('ply_number', 310, 'Point cloud example number.')
flags.DEFINE_integer('im_size', 512, 'Image size.')
flags.DEFINE_boolean('make_dataset', False, 'Iterate over whole dataset and save data.')
flags.DEFINE_boolean('correct_dataset', False, 'Correct wrong dataset examples.')
flags.DEFINE_boolean('add_bkg', False, 'Add background.')
def visualize(objs, ply_name, bb, skeleton=None, viewpoint="top", display=False, save=True, correct=False):
    scenario = ply_name + "_" + viewpoint
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=FLAGS.im_size, height=FLAGS.im_size, window_name=scenario, visible=display)
    vis.get_render_option().point_size = 1
    for obj in objs:
        vis.add_geometry(obj)
    ctr = vis.get_view_control()
    
    # ctr.set_up([0, -1, 0])
    # ctr.set_front([-1, 0, -0.1])
    # ctr.set_lookat(bb.get_center())
    
    parameters = o3d.io.read_pinhole_camera_parameters(os.path.join(FLAGS.params_dir, viewpoint + "-view-camera.json"))
    ctr.convert_from_pinhole_camera_parameters(parameters)
    K = parameters.intrinsic.intrinsic_matrix
    C = parameters.extrinsic
    f = parameters.intrinsic.get_focal_length()
    skeleton2D = []
    skeleton3D = []
    for joint in skeleton:
        center = joint.get_center()
        coord_3D = np.append(center,1)
        coord_2D_temp = np.dot(np.dot(np.append(K, [[0],[0],[0]], axis=1),C),coord_3D)
        coord_2D = [coord_2D_temp[0]/coord_2D_temp[2], coord_2D_temp[1]/coord_2D_temp[2]]
        skeleton2D.append(coord_2D)
        skeleton3D.append(center)
    skeleton2D = np.asarray(skeleton2D)
    skeleton3D = np.asarray(skeleton3D)

    if(save):
        save_render(vis, scenario, correct)
        save_depth(vis, scenario, correct)
        save_2D(skeleton2D, scenario, correct)
        save_3D(skeleton3D, scenario, correct)
        # save_cloud_top(objs[0], scenario)
    if(display):
        vis.run()
    vis.destroy_window()
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
def save_render(vis, scenario, correct):
    out_dir = FLAGS.out_render_dir
    if(correct):
        out_dir += "_correct"
    render = vis.capture_screen_image(os.path.join(out_dir, scenario + ".png"), do_render=True)
    # render = vis.capture_screen_float_buffer(do_render=True)
    # render = np.asarray(render)
    # cv2.imwrite(os.path.join(out_dir, "render_" + scenario + ".png"), render)
def save_depth(vis, scenario, correct):
    out_dir = FLAGS.out_depth_dir
    if(correct):
        out_dir += "_correct"
    depth = vis.capture_depth_float_buffer(do_render=True)
    depth_image = np.asarray(depth)
    depth_image[depth_image==0]=255
    cv2.imwrite(os.path.join(out_dir, scenario + ".png"), depth_image)
def save_2D(skeleton2D, scenario, correct):
    out_dir = FLAGS.out_2D_dir
    if(correct):
        out_dir += "_correct"
    np.save(os.path.join(out_dir, scenario + ".npy"), skeleton2D)
def save_3D(skeleton3D, scenario, correct):
    out_dir = FLAGS.out_3D_dir
    if(correct):
        out_dir += "_correct"
    np.save(os.path.join(out_dir, scenario + ".npy"), skeleton3D)
def save_cloud_top(cloud, scenario):
    o3d.io.write_point_cloud(os.path.join(FLAGS.out_clouds_top, scenario.replace("_front",'') + ".ply"), cloud)
def make_list():
    all_joints = sorted(os.listdir(FLAGS.joints_dir))
    all_clouds = []
    all_clouds_temp = sorted(os.listdir(FLAGS.clouds_dir))
    all_joints_indexes = [x.split(".")[0] for x in all_joints]
    for clouds in all_clouds_temp:
        if clouds.split(".")[0] in all_joints_indexes:
            all_clouds.append(clouds) 
    all_clouds_indexes = [x.split(".")[0] for x in all_clouds]
    return all_joints, all_clouds
def create_skeleton(joints, invert=1):
    skeleton = []
    for i, joint in enumerate(joints):
        # joints[i] = joint = [-joint[0], 2.5-joint[2], -joint[1]]
        joints[i] = joint = [invert*joint[0], invert*joint[1], invert*joint[2]]
        joint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3, resolution=10)
        joint_sphere.translate(joint, relative=False)
        skeleton.append(joint_sphere)
    return skeleton
def scale_center_all(cloud, skeleton, bb, mesh_frame, factor=100):
    center = bb.get_center()
    mesh_frame.scale(scale=factor, center=[0,0,0])
    bb.translate([-center[0], 0, -center[2]])
    cloud.translate([-center[0], 0, -center[2]])
    cloud.scale(scale=factor, center=[0,0,0])
    for joint_sphere in skeleton:
        joint_sphere.translate([-center[0], 0, -center[2]])
        joint_sphere.scale(scale=factor, center= [0,0,0])
    return cloud, skeleton
def decimate(pcd, quantity=5):
    return pcd.uniform_down_sample(every_k_points=quantity)
def remove_outliers(pcd, nb_neighbors=50, std_ratio=0.0001, nb_points=1000, radius=20):
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # display_inlier_outlier(pcd, ind)
    pcd, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    # display_inlier_outlier(pcd, ind)
    return pcd
def recompute_normals(pcd, radius=10, max_nn=30, orient=50):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn = max_nn))
    # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamRadius(radius = radius))
    # pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamKNN(knn = max_nn))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(orient)
    pcd_inv = o3d.geometry.PointCloud(pcd)
    pcd_inv.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_inv.normals))
    # pcd.orient_normals_towards_camera_location([0,10,0])
    return pcd, pcd_inv
def pcd2mesh(pcd, depth=9, scale=1.1):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale)
    mesh = mesh.merge_close_vertices(eps=0.01)
    mesh = mesh.compute_vertex_normals(normalized=True)
    mesh = mesh.compute_triangle_normals(normalized=True)
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.filter_smooth_taubin()
    mesh = mesh.subdivide_loop()
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh
def remove_invisible(pcd):
    diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    
    camera = [0, 1-diameter, 0]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    return pcd
def main(argv):
    bad = list(range(10, 281, 10))+ \
                list(range(4360, 4861, 10)) + \
                list(range(8920, 9661, 10)) + \
                list(range(13660, 14121, 10)) + \
                list(range(18160, 18921, 10)) + \
                list(range(22900, 23631, 10)) + \
                list(range(27650, 29401, 10)) + [420]
    all_joints, all_clouds = make_list()
    if(FLAGS.make_dataset):
        for joints_file, clouds_file in tqdm(zip(all_joints, all_clouds), total=len(all_joints)):
            f_joints = gzip.GzipFile(os.path.join(FLAGS.joints_dir,joints_file), "r")
            joints = np.load(f_joints)
            ply_name = clouds_file.split('.')[0]
            ply_number = int(ply_name.split('_')[1])
                
            # if ply_number in bad:
            # if ply_number in bad or os.path.isfile(os.path.join(FLAGS.out_render_dir, ply_name + "_front.png")):
            
            if os.path.isfile(os.path.join(FLAGS.out_clouds_top, ply_name + ".ply")):
                print("Existing")
                continue
            cloud = o3d.io.read_point_cloud(os.path.join(FLAGS.clouds_dir,clouds_file))
            # cloud.scale(scale=-1, center= [0,0,0])
            
            joints = np.delete(joints, 3, axis=1) #/ 100.
            skeleton = create_skeleton(joints, invert=1)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0,0,0])
            bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(joints))
            bb.scale(scale=2, center= bb.get_center())
            # o3d.visualization.draw_geometries([cloud] + skeleton + [mesh_frame] + [bb])
            cloud = cloud.crop(bb)
            # cloud = remove_outliers(cloud)
            # cloud, skeleton = scale_center_all(cloud, skeleton, bb, mesh_frame, factor=1)
            bb = cloud.get_axis_aligned_bounding_box()
            # o3d.visualization.draw_geometries([cloud] + skeleton + [mesh_frame] + [bb])
            # cloud, cloud_inv = recompute_normals(cloud)
            # mesh = pcd2mesh(cloud)
            # mesh_inv = pcd2mesh(cloud_inv)
            # visualize(objs=[bb] + [mesh_frame] + [mesh] + [mesh_inv] + skeleton, skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="top", display=True, save=False)
            # visualize(objs=[bb] + [mesh_frame] + [mesh] + [mesh_inv] + skeleton, skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="rear", display=True, save=False)
            # visualize(objs=[bb] + [mesh_frame] + [mesh] + [mesh_inv] + skeleton, skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="left", display=True, save=False)
            # visualize(objs=[bb] + [mesh_frame] + [mesh] + [mesh_inv] + skeleton, skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="right", display=True, save=False)
            # visualize(objs=[mesh] + [mesh_inv], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="top", display=False, save=True)
            # visualize(objs=[mesh] + [mesh_inv], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="rear", display=False, save=True)
            # visualize(objs=[mesh] + [mesh_inv], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="left", display=False, save=True)
            # visualize(objs=[mesh] + [mesh_inv], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="right", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="front", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="rear", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="left", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="right", display=False, save=True)
            
            # try:
            #     cloud = remove_invisible(cloud)
            # except:
            #     pass
            visualize(objs=[cloud] + skeleton + [mesh_frame], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="top", display=True, save=False)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="rear", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="left", display=False, save=True)
            # visualize(objs=[cloud], skeleton=skeleton, ply_name=ply_name, bb=bb, viewpoint="right", display=False, save=True)
            # quit(0)
        
if __name__ == '__main__':
    app.run(main)