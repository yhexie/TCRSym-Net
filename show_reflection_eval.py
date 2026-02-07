import os
import numpy as np
import io_ply
from tqdm import tqdm
import open3d as o3d
import utility
from tools.plane import drawPlane
##-----------------------------------------
###说明：原始点云和标注文件用o3的读取并可视化
###-----------------------------------------
root_train_aug = '../data/compare'
root_eval = './result'
pcds = sorted(os.listdir(root_train_aug)) #获取文件夹下所有的文件和文件夹名称
pcds = [pcd for pcd in pcds if pcd[-4:]=='.ply']
pcds_split = [room for room in pcds if not 'model_{}'.format(5) in room]
N_points = 24096

for roof_name in tqdm(pcds_split, total=len(pcds_split)):
    roof_path = os.path.join(root_train_aug, roof_name)
    pcd = o3d.io.read_point_cloud(roof_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(points.shape)
    center = points[np.random.choice(N_points)][:3]
    block_size = 100.0
    block_min = center - [block_size / 2.0, block_size / 2.0, 0]
    block_max = center + [block_size / 2.0, block_size / 2.0, 0]
    point_idxs = \
    np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
            points[:, 1] <= block_max[1]))[0]

    if point_idxs.size >= N_points:
        selected_point_idxs = np.random.choice(point_idxs, N_points, replace=False)
    else:
        selected_point_idxs = np.random.choice(point_idxs, N_points, replace=True)

    # normalize
    selected_points = points[selected_point_idxs, :]
    selected_colors = colors[selected_point_idxs, :]

    coord_min, coord_max = np.amin(selected_points, axis=0)[:3], np.amax(selected_points, axis=0)[:3]
    length_x= coord_max - coord_min

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(selected_points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(selected_colors))
    sym_file = roof_name[:-4] + '.eval.sym'
    sym_path = os.path.join(root_eval, sym_file)
    model_s = np.loadtxt(sym_path)  # 5x3
    syms = model_s[1:, :]  # 4x3，第二行开始
    check_ = np.zeros((4, 3))
    check_sym = (syms != check_)
    nozero = np.nonzero(check_sym)
    row_id = nozero[0]  # 0,1,2三行存放的是镜面对称，3存放的是旋转轴
    row_id = utility.remove_duplicates_with_dict(row_id)
    row_id = np.array(row_id)
    target_mode = 0
    center = model_s[0, :]  # 重要

    if row_id.shape[0] == 1:
        if row_id[-1] == 3:
            target_mode = 1  # 只有1个信息，存放在3，旋转对称
        else:
            target_mode = 0  # 只有1个信息，存放在非3，平面对称
        multi_s = syms[row_id]  ##存放的是对称平面法向量？---------------
    elif row_id.shape[0] == 2:
        if row_id[-1] == 3:
            target_mode = 2  # 有2个信息，存放在3，旋转对称
            multi_s = syms[row_id[0]]  # 奇怪？？？只取一个
        else:
            target_mode = 0
            multi_s = syms[row_id]
    elif row_id.shape[0] == 3:
        target_mode = 0  # 有3个信息，存放在0，1，2，平面对称
        multi_s = syms[row_id]
    else:
        target_mode = 0
        multi_s = syms[row_id]

    multi_s_point = multi_s + center

    # 生成法向量线段的端点坐标
    normal_vector = multi_s[0]
    normal_vector1 = multi_s[1]
    normal_vector2 = multi_s[2]
    # 求平面上的一点
    point_in_plane = center

    # 沿着法向量方向上的另一点
    endpoint = point_in_plane + normal_vector * 5
    theta = np.arctan2(normal_vector[1],normal_vector[0])
    theta1 = np.arctan2(normal_vector1[1], normal_vector1[0])
    theta2 = np.arctan2(normal_vector2[1], normal_vector2[0])
    mesh_plane = drawPlane(10,theta,True)
    # 创建法向量线段
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([point_in_plane, endpoint])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    mesh_plane = drawPlane(point_in_plane, length_x[1], theta, True)
    mesh_plane1 = drawPlane(point_in_plane, length_x[0], theta1, True)
    mesh_plane2 = drawPlane(point_in_plane, length_x[0], theta2, True)
    num = len(mesh_plane.vertices)
    print(num)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.3333, 0.66669, 1.0, 0.5]  # Green with alpha=0.5

    material2 = o3d.visualization.rendering.MaterialRecord()
    material2.shader = "defaultLit"
    material2.base_color = [1.0, 0.66669, 0.498, 0.5]
    # 可视化结果
    #o3d.visualization.draw_geometries([point_cloud, mesh_plane, mesh_plane2,  mesh_frame], mesh_show_back_face=True)
    o3d.visualization.draw( [{'name': 'point', 'geometry': point_cloud},{'name': 'box', 'geometry': mesh_plane, 'material': material} ,
                             {'name': 'box1', 'geometry': mesh_plane1, 'material': material},
                             {'name': 'box2', 'geometry': mesh_plane2, 'material': material2}], bg_color=[1,1,1,1],show_skybox = False)
    sym_plane_path = os.path.join(root_train_aug, roof_name[:-4] + '_plane_pointnet.stl')
    o3d.io.write_triangle_mesh(sym_plane_path, mesh_plane)
    sym_plane_path1 = os.path.join(root_train_aug, roof_name[:-4] + '_plane_pointnet1.stl')
    o3d.io.write_triangle_mesh(sym_plane_path1, mesh_plane1)
    sym_plane_path2 = os.path.join(root_train_aug, roof_name[:-4] + '_plane_pointnet2.stl')
    o3d.io.write_triangle_mesh(sym_plane_path2, mesh_plane2)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(point_cloud)
    # vis.add_geometry(mesh_plane)
    # vis.add_geometry(mesh_plane2)
    # render_option = vis.get_render_option()
    # render_option.point_size = 4
    # render_option.background_color = np.asarray([1, 1, 1])
    # # 设置渲染选项以显示背面
    # render_option.mesh_show_back_face = True
    # ctr = vis.get_view_control()
    #
    # # 修改相机位置（这将间接影响光源的位置）
    # # 例如，将相机向前移动一些距离
    # ctr.set_front([10, 0, -1])  # 修改前方向向量
    # ctr.set_lookat(point_in_plane)  # 修改观察点
    # ctr.set_up([0, 0, 1])  # 修改上方向向量
    # ctr.convert_to_pinhole_camera_parameters()  # 确保使用针孔相机模型
    #
    # # 更新视图
    # vis.update_renderer()
    # #
    # vis.run()
    # vis.destroy_window()


