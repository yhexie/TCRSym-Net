import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from https://github.com/qinglew/PointCloudTransformer
# Changed the method used for doing furthest_point_sample
# to avoid installing pointnet2_ops_lib
def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def furthest_point_sample(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :]  # (B, 3)
        # cur_dist -> B, N
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz)).reshape(B, N)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


def cal_loss(pred, ground_truth, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]

    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]

    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]

    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)  # [B, s, n]
    grouped_features = index_points(features, idx)  # [B, s, n, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)],
                                    dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]

    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)  # [B, s, 3]
    new_features = index_points(features, fps_idx)  # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)  # [B, s, k]
    grouped_features = index_points(features, idx)  # [B, s, k, D]

    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)],
                                    dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]


class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s,k, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s
        self.k = k
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)  # (B, N, in_channels//2)
        #s为采样点数？？
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=self.k, coords=coords,
                                                    features=x)  # [B, s, 3], [B, s, 32, in_channels] 进去是N‘*64，出来时N’*k*128
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)  # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))  # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels] 可能的值N’*128  N’*256
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)  # [B, in_channels, s]
        return new_xyz, new_feature


class OA(nn.Module):
    """
    Offset-Attention Module.
    """

    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]

        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

# class Swp1d(torch.nn.Module):
#     def __init__(self, bs, in_features, out_features, bias=False):
#         super(Swp1d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = torch.nn.Parameter(torch.Tensor(bs, in_features, out_features))
#         if bias:
#             self.bias = torch.nn.Parameter(torch.Tensor(in_features))
#         else:
#             self.register_parameter('bias', None)
#         #random weights drawn from Gaussian distributions with a fixed standard deviation of 0.005.
#         # self.reset_parameters()
#         torch.nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
#     # def reset_parameters(self):
#     #     stdv = 1.0 / math.sqrt(self.weight.size())
#     #     for weight in self.parameters():
#     #         weight.data.uniform_(-stdv, stdv)
#     def forward(self, input):
#         # input = torch.pow(input, 2)
#         y = torch.bmm(input, self.weight)
#         return y

#通过全连接到高维嵌入空间Embeding
class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512,256,128],K=[16,32,64]):
        super(NeighborEmbedding, self).__init__()
        #self.conv0 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        #self.bn0 = nn.BatchNorm1d(32)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        #self.sg0 = SG(s=samples[0], k=K[0], in_channels=64, out_channels=64)
        #self.sg1 = SG(s=samples[1], k=K[1], in_channels=128, out_channels=128)
        #self.sg2 = SG(s=samples[0], k=K[1], in_channels=256, out_channels=256)

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]
        features = F.relu(self.conv1(x))  # [B, 64, N]
        #features1 = F.relu(self.bn1(self.conv1(features)))  # [B, 32, N]
        #xyz1, features1 = self.sg0(features, xyz)  # [B, 64, 1024]

        features1 = F.relu(self.conv2(features))  # [B, 128, N]))) 
        features3 = F.relu(self.conv3(features1)) # [B, 256, N]
        #xyz2, features2 = self.sg1(features1, xyz1)  # [B, 128, 512]
        #_, features3 = self.sg2(features1, xyz)  # [B, 256, 256]

        return features3


class PCT(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        #将采样分散到嵌入空间
        samples = [512, 256, 128]
        self.neighbor_embedding = NeighborEmbedding(samples)

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)
        #self.swp = Swp1d(1, self.num_points, 1)  # 注意这块和批处理的大小一致

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            #nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.neighbor_embedding(x)

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.linear(x)
        # swp_x = self.swp(x)
        # swp_x = swp_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        # x_out = torch.cat([swp_x], 1)
        #
        x_max = torch.max(x, dim=-1).values
        x_mean = torch.mean(x, dim=-1)

        #新增的，这个将1个1024列向量复制了4096份，注意这种操作是否存在啥问题
        x_max = x_max.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return x_max #x_out #torch.cat([x_max, x_mean], dim=-1)

class AcientSymNetPCT(nn.Module):
    def __init__(self, num_points):
        super(AcientSymNetPCT, self).__init__()
        self.num_points = num_points
        self.encoder = PCT( self.num_points )
        self.conv1_cent = torch.nn.Conv1d(1024, 640, 1)
        self.conv1_self1 = torch.nn.Conv1d(1024, 640, 1)
        self.conv1_self2 = torch.nn.Conv1d(1024, 640, 1)
        self.conv1_self3 = torch.nn.Conv1d(1024, 640, 1)
        self.conv1_choose = torch.nn.Conv1d(1024, 640, 1)
        self.conv1_mode = torch.nn.Conv1d(1024, 640, 1)

        self.conv2_cent = torch.nn.Conv1d(640, 256, 1)
        self.conv2_self1 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_self2 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_self3 = torch.nn.Conv1d(640, 256, 1)
        self.conv2_choose = torch.nn.Conv1d(640, 256, 1)
        self.conv2_mode = torch.nn.Conv1d(640, 256, 1)

        self.conv3_cent = torch.nn.Conv1d(256, 128, 1)
        self.conv3_self1 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_self2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_self3 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_choose = torch.nn.Conv1d(256, 128, 1)
        self.conv3_mode = torch.nn.Conv1d(256, 128, 1)

        self.conv4_cent = torch.nn.Conv1d(128, 3, 1)
        self.conv4_self1 = torch.nn.Conv1d(128, 9, 1)
        self.conv4_self2 = torch.nn.Conv1d(128, 9, 1)
        self.conv4_self3 = torch.nn.Conv1d(128, 3, 1)
        self.conv4_choose = torch.nn.Conv1d(128, 3, 1)
        self.conv4_mode = torch.nn.Conv1d(128, 3, 1)

    def forward(self, x):
        bs = 1
        batch_size = x.shape[0]
        ap_x = x.transpose(2, 1).contiguous()  # [1, 6, 4096]
        #print(ap_x.shape)
        ap_x = ap_x.float()
        ap_x = self.encoder(ap_x)
        #print(ap_x.shape) #[1, 1024, 4096]
        cent_x = F.relu(self.conv1_cent(ap_x))
        self1_x = F.relu(self.conv1_self1(ap_x))
        self2_x = F.relu(self.conv1_self2(ap_x))
        self3_x = F.relu(self.conv1_self3(ap_x))
        choose_x = F.relu(self.conv1_choose(ap_x))
        mode_x = F.relu(self.conv1_mode(ap_x))

        cent_x = F.relu(self.conv2_cent(cent_x))
        self1_x = F.relu(self.conv2_self1(self1_x))
        self2_x = F.relu(self.conv2_self2(self2_x))
        self3_x = F.relu(self.conv2_self3(self3_x))
        choose_x = F.relu(self.conv2_choose(choose_x))
        mode_x = F.relu(self.conv2_mode(mode_x))

        cent_x = F.relu(self.conv3_cent(cent_x))
        self1_x = F.relu(self.conv3_self1(self1_x))
        self2_x = F.relu(self.conv3_self2(self2_x))
        self3_x = F.relu(self.conv3_self3(self3_x))
        choose_x = F.relu(self.conv3_choose(choose_x))
        mode_x = F.relu(self.conv3_mode(mode_x))

        cent_x = self.conv4_cent(cent_x).view(bs, 3, self.num_points)
        self1_x = self.conv4_self1(self1_x).view(bs, 9, self.num_points)
        self2_x = self.conv4_self2(self2_x).view(bs, 9, self.num_points)
        self3_x = self.conv4_self3(self3_x).view(bs, 3, self.num_points)
        choose_x = torch.sigmoid(self.conv4_choose(choose_x)).view(bs, 3, self.num_points)
        mode_x = torch.sigmoid(self.conv4_mode(mode_x)).view(bs, 3, self.num_points)

        out_cent = cent_x.contiguous().transpose(2, 1).contiguous()  # 中心点，平面上1个点
        out_self1 = self1_x.contiguous().transpose(2, 1).contiguous()  # 3 possible reflection 3个可能得对称平面
        out_self2 = self2_x.contiguous().transpose(2, 1).contiguous()  # 3 possible foot point 3个可能得足点
        out_self3 = self3_x.contiguous().transpose(2, 1).contiguous()  # foot point, axis, circle point #1个点？

        out_choose = choose_x.contiguous().transpose(2, 1).contiguous()  # 对称对应点？？
        out_mode = mode_x.contiguous().transpose(2, 1).contiguous()  # 对称模式？
        out_ref = out_self1
        out_foot_ref = out_self2
        out_rot = out_self3
        return out_cent, out_ref, out_foot_ref, out_rot, out_choose, out_mode



if __name__ == '__main__':
    pc = torch.rand(2, 3, 122)

    # testing encoder
    encoder = PCT(122)
    out = encoder.forward(pc)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(out[1].shape)
    print(total_params)