from torch.nn.modules.loss import _Loss
import torch
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
def loss_calculation( pred_cent, pred_reflection ,pred_foot_ref, pred_rot, pred_num, pred_mode,
                     target_s, points, w, target_mode):
    #需要把center0加到预测的pred_ref上吗？
    bs = 1
    num_p = 4096
    points = points.contiguous().view(bs * num_p, 1, 3)  # 4096*1*3 输入点云
    pred_num = pred_num.contiguous().view(bs * num_p, 3)
    pred_mode = pred_mode.contiguous().view(bs * num_p, 3) #预测的模式？
    pred_cent = pred_cent.contiguous().view(bs * num_p, 1, 3)
    pred_reflection = pred_reflection.contiguous().view(bs * num_p, -1, 3) #3个点,预测的法向量？？？？------------
    pred_foot_ref = pred_foot_ref.contiguous().view(bs * num_p, -1, 3) #足点偏移？d的向量
    pred_rot = pred_rot.contiguous().view(bs * num_p, -1, 3)
    pred_rot_foot = pred_rot.view(bs * num_p, -1, 3)

    target_mode = target_mode.view(-1)  #对称模式？
    target_mode_ = target_mode
    target_mode = target_mode.view(bs, 1, 1).repeat(1, num_p, 1).view(bs * num_p) #重复1000次

    target_s = target_s.view(bs, 1, -1, 3).repeat(1, num_p, 1, 1).view(bs * num_p, -1, 3) #重复1000次
    target_cent = target_s[:, 0, :].view(bs * num_p, -1, 3) # 轴上一点？对称平面上一点？任意一点，复制1000次
    target_sym = target_s[:, 1:, :].view(bs * num_p, -1, 3)  #中心点+法向量
    target_sym_vec = torch.add(target_sym, -target_cent)  # 1000,-1 ,3 法向量
    # target_sym_vec = torch.norm(target_sym_vec,dim=2)
    target_sym_vec = F.normalize(target_sym_vec,p=2,dim=2)

    cent_pred = torch.add(points, pred_cent)  # 1000,1,3
    ref_pred = torch.add(points, pred_reflection)
    ref_foot_pred = torch.add(points, pred_foot_ref) #点加上足点偏移，得到 足点

    cross_entropy = torch.nn.CrossEntropyLoss()  #交叉熵损失？
    mode_loss = cross_entropy(pred_mode, target_mode)#对称类型的损失L_type
    #这个center_loss很奇怪，对称平面点？轴上点？
    center_loss = torch.mean(torch.norm((cent_pred - target_cent), dim=2), dim=1)  # (1000)

######### cost matrix
######### cosine angle of pred norm and target norm
    mean_pred_refection = torch.mean(pred_reflection, dim=0)
    mean_ref_pred = torch.mean(ref_pred, dim=0)
    mean_target_vec = torch.mean(target_sym_vec, dim=0)
    cost_matrix = torch.zeros(mean_pred_refection.shape[0], target_sym_vec.shape[1])

    for i in range(mean_pred_refection.shape[0]):
        for j in range(mean_target_vec.shape[0]):
            a = mean_pred_refection[i, :].view(1, 3).float()
            b = mean_target_vec[j, :].view(3, 1).float()
            product = torch.mm(a, b)
            norm_a = torch.norm(a, dim=1)
            norm_b = torch.norm(b, dim=0)
            cost = torch.abs(product / (torch.add(norm_a, 0.00001)*torch.add(norm_b, 0.00001)))
            cost_matrix[i, j] = torch.acos(cost.reshape(-1))

###### optimal assiment
###### min cost for each point is the point-wise solusion
    row_id_, col_id_ = linear_sum_assignment(cost_matrix.detach().numpy())
    if mean_target_vec.shape[0] > 1:
        corr = np.array([row_id_,col_id_]).T
        ordered_id = corr[corr[:,1].argsort()]
        row_id = ordered_id[:,0]
        col_id = ordered_id[:,1]
    else :
        row_id = row_id_
        col_id = col_id_
    ref_out = ref_pred[:, row_id, :] #预测对称点
    ref_out_vec = pred_reflection[:, row_id, :] #预测法向量

    ref_out_foot = ref_foot_pred[:, row_id, :] #足点
    ref_out_vec_foot = pred_foot_ref[:, row_id, :] #足点偏移 D
    #标签数组[1, 1, 1],[1, 1, 0]，依次
    target_id = label_trans(torch.tensor(row_id)).cuda().float()
    #-----------重要的------------------
    target_ref = ref_pt(points, target_cent, target_sym_vec)[:, col_id, :].cuda() #--根据对称平面法向量求对称点吗？背景值对称点
    target_foot_ref = points + 0.5*(target_ref-points) #足点
    target_sym_vec_orderd = target_sym_vec[:,col_id,:]

    id_loss = torch.nn.BCELoss() #计算目标值和预测值之间的二进制交叉熵损失函数
    mean_pred_num = torch.mean(pred_num, dim=0)
    num_loss = id_loss(mean_pred_num, target_id)   # (1)法向量数目损失？

    ref_loss = 0
    ref_foot_loss = 0
    ref_co_loss = 0
    rot_foot_loss = 0
    rot_co_loss = 0
    if target_mode_ != 0:#旋转对称？
        rot_foot_pred = torch.add(points, pred_rot_foot)#1000,1,3
        point_to_cent = torch.add(-points, target_cent)#1000,1,3
        #torch.bmm是加了batch的矩阵乘法
        product = torch.bmm(target_sym_vec.view(num_p,1,3), point_to_cent.view(num_p,3,1)).view(num_p)
        cos = product / (torch.norm(point_to_cent.view(num_p, 3), dim=1) *
                         torch.norm(target_sym_vec.view(num_p,3), dim=1)+0.00001).view(num_p)
        point_to_cent_nom = torch.norm(point_to_cent.view(num_p,3), dim=1)
        cent_to_foot = (-point_to_cent_nom * cos).view(num_p,1).repeat(1,3)*(target_sym_vec.view(num_p,3))
        target_rot_foot = target_cent + cent_to_foot.view(num_p,1,3)
        rot_foot_loss = torch.mean(torch.norm(target_rot_foot - rot_foot_pred, dim=2), dim=1).cuda() #0.1
        pt_to_foot = rot_foot_pred - points
        rot_co_loss = torch.mean(torch.bmm(pt_to_foot.view(num_p,1,3), cent_to_foot.view(num_p,3,1)).view(-1)).cuda()**(2)#0.001

    if target_mode_  != 1:#镜面对称？
        ref_out_len = torch.norm(ref_out_vec, dim=2)
        ref_distance = torch.norm((ref_out - target_ref), dim=2) #对称点损失
        ref_loss = torch.mean(torch.div(ref_distance, ref_out_len+0.00001), dim=1).cuda()
        ref_foot_loss = torch.mean(torch.norm(ref_out_foot - target_foot_ref, dim=2), dim=1).cuda()#0.1
        ref_co_loss = torch.mean(torch.mean(torch.norm(ref_out_vec_foot * 2 - pred_reflection[:, row_id, :], dim=2), dim=1)).cuda()**(2)#0.1

#######caculate angle error
    if target_mode_ == 1:
        pred_axis = cent_pred.view(num_p,3) - rot_foot_pred.view(num_p,3)
        best_norm = F.normalize(pred_axis,p=2,dim=1).view(num_p,1,3)
        target_norm = target_sym_vec_orderd[0, :].view(1, 3, 1).repeat(num_p,1,1)
        products = torch.abs(torch.bmm(best_norm, target_norm))
    else:
        best_ref = torch.mean(ref_out_vec, dim=0)
        products = torch.zeros(best_ref.shape[0])
        for i in range(best_ref.shape[0]):
            best_norm = best_ref[i, :].view(1, 3).cuda().float()
            target_norm = target_sym_vec_orderd[0, i, :].view(3, 1).float()
            product = torch.abs(torch.mm(best_norm, target_norm) / (
                        torch.norm(best_norm, dim=1) * torch.norm(target_norm.contiguous().transpose(1, 0), dim=1))+0.00001)
            products[i] = product

    dis = torch.mean(w * center_loss + ref_loss + ref_foot_loss + rot_foot_loss, dim=0)
    loss = dis + 2 * num_loss + mode_loss + w * 0.5*ref_co_loss + 0.5* w * rot_co_loss

    center_dis = torch.mean(center_loss.view(bs, num_p), dim=1)

    ref_dis = dis
    angle_error = torch.mean(torch.acos(products) / math.pi * 180)

    error_num = torch.mean(num_loss)
    error_mode = torch.mean(mode_loss)

    return loss, loss, center_dis.data.cpu(), ref_dis, angle_error, error_num.data.cpu(), error_mode.cpu()

#计算对称点
def ref_pt(pt, cent, sym_vect):
    pt_pred = torch.zeros(sym_vect.shape)
    bs = 1
    for i in range(sym_vect.shape[1]):
        center = cent.view(bs * 4096,3,1) #对称平面上点
        norm = sym_vect[:, i, :].view(bs * 4096,1,3) #法向量
        d = -torch.bmm(norm,center) #点到对称平面距离
        pt_ = pt-2*(torch.bmm(norm, pt.view(bs * 4096,3,1)) + d)*norm #对称点
        pt_pred[:, i, :] = pt_.view(bs * 4096,3)
    return pt_pred

def label_trans(input):
    if input.shape[0] == 3:
        label = torch.tensor([1, 1, 1])
    elif input.shape[0] == 2:
        if input.equal(torch.tensor([0, 1])) or input.equal(torch.tensor([1, 0])):
            label = torch.tensor([1, 1, 0])

        if input.equal(torch.tensor([0, 2])) or input.equal(torch.tensor([2, 0])):
            label = torch.tensor([1, 0, 1])

        if input.equal(torch.tensor([1, 2])) or input.equal(torch.tensor([2, 1])):
            label = torch.tensor([0, 1, 1])
    else: #input.shape[0] == 1:
        if input.equal(torch.tensor([0])):
            label = torch.tensor([1, 0, 0])

        if input.equal(torch.tensor([1])):
            label = torch.tensor([0, 1, 0])

        if input.equal(torch.tensor([2])):
            label = torch.tensor([0, 0, 1])
        else:
            try:
                return label
            except Exception as e:
                print(e.args)
                print(input)
    return label

class Loss(_Loss):

    def __init__(self, num_points_mesh):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh

    def forward(self, pred_cent, pred_reflection,  pred_foot_ref, pred_rot,pred_num, pred_mode,
                            target_s, points, w, target_mode):
        return loss_calculation(pred_cent, pred_reflection, pred_foot_ref, pred_rot,pred_num, pred_mode,
                            target_s, points, w, target_mode)
