import numpy as np
import time
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import open3d as o3d
from PCT import AcientSymNetPCT
import utility
from numpy import linalg as LA
import sklearn.cluster as skc
from tools.plane import drawPlane
def eval():
    ckpt = "./results/model_pct.pth"
    eval_path = 'eval/outside'
    root_eval = './data/outside'
    N_points = 4096
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    ckpt = torch.load(str(ckpt), map_location='cpu')
    model = AcientSymNetPCT(N_points)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print("开始进行预测：")

    pcds = sorted(os.listdir(root_eval))  # 获取文件夹下所有的文件和文件夹名称
    pcds = [pcd for pcd in pcds if pcd[-4:] == '.ply']
    pcds_split = [room for room in pcds if not 'model_{}'.format(5) in room]

    for roof_name in tqdm(pcds_split, total=len(pcds_split)):
        roof_path = os.path.join(root_eval, roof_name)
        # 读取点云
        pcd = o3d.io.read_point_cloud(roof_path)
        points = np.asarray(pcd.points)

        print(points.shape)
        center = points[np.random.choice(N_points)][:3]
        block_size = 200.0
        block_min = center - [block_size / 2.0, block_size / 2.0, 0]
        block_max = center + [block_size / 2.0, block_size / 2.0, 0]
        point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]

        if point_idxs.size >= N_points:
            selected_point_idxs = np.random.choice(point_idxs, N_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, N_points, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(selected_points))
        # 可视化点云
        #o3d.visualization.draw_geometries([point_cloud])

        norm_points, center0, scale = utility.normalize2(selected_points, unit_ball=True)
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud1.points = o3d.utility.Vector3dVector(np.array(norm_points))

        current_points = norm_points[None]
        current_points = torch.Tensor(current_points)
        center0 = torch.Tensor(center0)
        current_points, center0 = current_points.to(device),center0.to(device)
        with torch.no_grad():
            pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode = model(current_points)
            #pred_cent = pred_cent * scale + center0
            pred_cent = pred_cent.detach().cpu().data.numpy()
            pred_reflection = pred_ref.detach().cpu().data.numpy()
            pred_foot_ref = pred_foot_ref.detach().cpu().data.numpy()

            current_points = current_points.view(N_points, 3)
            current_points = current_points.detach().cpu().data.numpy()
            pred_cent = pred_cent.reshape(N_points, 3)
            cent_pred = (current_points + pred_cent)

            pred_reflection = pred_reflection.reshape(N_points, -1, 3)
            my_sym = pred_reflection
            my_norm = np.zeros(my_sym.shape)

            for i in range(my_sym.shape[1]):
                for k in range(3):
                    my_norm[:, i, k] = my_sym[:, i, k] / np.linalg.norm(my_sym[:, i, :], axis=1)

            mean_norm = np.mean(my_norm, axis=0)  # n*3
            mean_cent = np.mean(cent_pred, axis=0)  # 1*3
            out_cent = mean_cent

            out_sym = np.zeros(mean_norm.shape)
            sym_conf = np.zeros(mean_norm.shape[0])
            norm_conf_list = np.zeros(mean_norm.shape[0])
            for i in range(my_norm.shape[1]):
                this_norm = my_norm[:, i, :].reshape(N_points, 3)
                dim_conf = 0
                for t in range(3):
                    this_dim = this_norm[:, t].reshape(N_points, 1)
                    # target_dim = target_sym[i,j]
                    mean_dim = np.mean(this_dim, axis=0)
                    db = skc.DBSCAN(eps=0.2, min_samples=500).fit(this_dim)
                    labels = db.labels_
                    clster_center = np.mean(this_dim[labels[:] == 0], axis=0)
                    out_sym[i, t] = clster_center
                    dim_conf += len(labels[labels[:] == 0]) / len(labels)

                if np.isnan(out_sym[i]).any():
                    norm_conf = 0
                else:
                    norm_conf = dim_conf / 3
                    # norm_conf = 1
                norm_conf_list[i] = norm_conf
            my_ref = utility.reflect(norm_points, out_cent, out_sym)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

            normal_vector = out_sym[0]
            # 求平面上的一点
            out_cent = out_cent * scale + center0.detach().cpu().data.numpy()
            point_in_plane = out_cent
            endpoint = point_in_plane + normal_vector * 5
            # 创建法向量线段
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([point_in_plane, endpoint])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])

            theta = np.arctan2(normal_vector[1], normal_vector[0])
            mesh_plane = drawPlane(point_in_plane, scale, theta, True)
            # 可视化结果
            o3d.visualization.draw_geometries([point_cloud,point_cloud1,mesh_plane,mesh_frame],mesh_show_back_face=True)

            if not os.path.exists(eval_path):
                os.makedirs(eval_path)
            #print(room_name)
            f_Edges = eval_path + '/{}{}{}'.format(roof_name[:-3], 'eval_pct', '.sym')
            with open(f_Edges, 'w') as file:
                # 写入一条记录
                #mag = LA.norm(pred_ref[0][0])
                #pred_normal = pred_ref[0][0] / mag
                # out_cent = out_cent * scale + center0.detach().cpu().data.numpy()
                #file.writelines('{} {} {}\n'.format(center0[0], center0[1], center0[2]))
                file.writelines('{} {} {}\n'.format(out_cent[0],out_cent[1],out_cent[2]))
                file.writelines('{} {} {}\n'.format(out_sym[0][0], out_sym[0][1], out_sym[0][2]))
                file.writelines('{} {} {}\n'.format(out_sym[1][0], out_sym[1][1], out_sym[1][2]))
                file.writelines('{} {} {}\n'.format(out_sym[2][0], out_sym[2][1], out_sym[2][2]))
                file.writelines('{} {} {}\n'.format(0.0000, 0.0000, 0.0000))

if __name__ == '__main__':
    eval()