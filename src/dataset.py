import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import Dataset
#import  io_ply
import utility

class AcientDataset(Dataset):
    def __init__(self, split='train', data_root='./data/', num_point=4096, block_size=20.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        if split == 'train':
            self.root_train = data_root + 'train_aug/'
            print(self.root_train)
            self.sym_dir = data_root + 'train_aug/'  # 对称信息？标注信息？
            pcds = sorted(os.listdir(self.root_train))  # 获取文件夹下所有的文件和文件夹名称
            pcds = [pcd for pcd in pcds if pcd[-4:] == '.ply']
            self.pcds_split = [pcd for pcd in pcds if not 'model_{}'.format(5) in pcd]
        else:
            self.root_train = data_root + 'test_aug/'#'roof300/funspointcloud/'#'test_aug/'
            print(self.root_train)
            self.sym_dir = data_root + 'test_aug/'#'roof300/funspointcloud/' #'test_aug/' # 对称信息？标注信息？
            pcds = sorted(os.listdir(self.root_train))  # 获取文件夹下所有的文件和文件夹名称
            pcds = [pcd for pcd in pcds if pcd[-4:] == '.ply']
            self.pcds_split = [pcd for pcd in pcds if not 'model_{}'.format(5) in pcd]

        self.roof_points= []
        self.roof_coord_min, self.roof_coord_max = [], []

        for roof_name in tqdm(self.pcds_split, total=len(self.pcds_split)):
            roof_path = os.path.join( self.root_train, roof_name)
            #print(roof_path)
            #roof_data = np.load(room_path)  # x y z cx cy cz r, N*7
            #roof_data = io_ply.read_ply(room_path)
            pcd = o3d.io.read_point_cloud(roof_path)
            points = pcd.points
            # print(points.shape)# 输出数组的形状（行列数）
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.roof_points.append(points)
            self.roof_coord_min.append(coord_min), self.roof_coord_max.append(coord_max)
        self.rooms_count = len(self.pcds_split)

    def __getitem__(self, idx):
        point_A= self.roof_points[idx]   # N * 3
        points= np.asarray(point_A)  # A已经变成n*3的矩阵
        N_points = self.num_point
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            #注意这里，进行了范围采样1*1m的方形区域？
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        #随机采样4096个点
        if point_idxs.size >= N_points:
            selected_point_idxs = np.random.choice(point_idxs, N_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, N_points, replace=True)

        selected_points = points[selected_point_idxs, :]  # N_points * 3
        # normalize归一化，预测的结果*scale+center
        norm_points, center0, scale = utility.normalize2(selected_points,unit_ball=True)
        current_points = np.zeros((N_points, 6))  # N_points * 3
        #current_points[:, 0:3] = norm_points
        if self.transform is not None:
            current_points = self.transform(current_points)

        ##------------------加载标注文件-------------------------------
        sys_dir = self.root_train
        sym_file = self.pcds_split[idx][:-4] + '.sym'
        sym_path = os.path.join(sys_dir, sym_file)
        model_s = np.loadtxt(sym_path)  # 5x3

        syms = model_s[1:, :]  # 4x3，第二行开始
        check_ = np.zeros((4, 3))
        check_sym = (syms != check_)
        nozero = np.nonzero(check_sym)
        row_id = nozero[0]  # 0,1,2三行存放的是镜面对称，3存放的是旋转轴
        row_id = utility.remove_duplicates_with_dict(row_id)
        row_id = np.array(row_id)
        target_mode = 0
        center_recenter = (model_s[0, :] - center0) / scale
        #center = model_s[0, :]  # 重要

        if row_id.shape[0] == 1:
            if row_id[-1] == 3:
                target_mode = 1
            else:
                target_mode = 0
            multi_s = syms[row_id]
        elif row_id.shape[0] == 2:
            if row_id[-1] == 3:
                target_mode = 2
                multi_s = syms[row_id[0]]
            else:
                target_mode = 0
                multi_s = syms[row_id]
        elif row_id.shape[0] == 3:
            target_mode = 0
            multi_s = syms[row_id]
        else:
            target_mode = 0
            multi_s = syms[row_id]

        multi_s_point = multi_s + center_recenter
        multi_s = np.vstack([center_recenter, multi_s_point])
        target_s = multi_s
        target_num = target_s.shape[0] - 1
        return norm_points,center0,scale, target_mode,target_s,target_num

    def __len__(self):
        return self.rooms_count



if __name__ == '__main__':
    data_root = '../data/'
    num_point, block_size, sample_rate = 4096, 10.0, 0.01

    point_data = AcientDataset(split='original', data_root=data_root, num_point=num_point,  block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
    for idx in range(4):
        end = time.time()
        for i, input in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()