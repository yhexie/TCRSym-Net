import argparse
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset import AcientDataset
from model import AcientSymNet
from loss import Loss
from utils import plot_curve
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'acient_ds', help='shapenet or scan2cad')
parser.add_argument('--dataset_root', type=str, default = './data/')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--nepoch', type=int, default=100, help='max number of epochs to original')
parser.add_argument('--lr', default=0.00005, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=1, help='learning rate')
parser.add_argument('--w_rate', default=0.9, help='learning rate decay rate')
opt = parser.parse_args()
def main():
    savel_model_path = './results'
    opt.num_points = 4096
    my_logger = logging.Logger('first_logger')
    my_handler = logging.FileHandler('log_simplePointNet.log')
    my_handler.setLevel(logging.INFO)
    my_format = logging.Formatter("%(message)s")
    my_handler.setFormatter(my_format)
    my_logger.addHandler(my_handler)
    # 训练------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(savel_model_path):
        os.makedirs(savel_model_path)
    model = AcientSymNet(num_points=opt.num_points)
    model = model.to(device)
    try:
        checkpoint = torch.load(str(savel_model_path) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        start_epoch = 0

    opt.decay_start = False
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, -1)
    opt.w *= opt.w_rate

    train_dataset = AcientDataset('train', opt.dataset_root, opt.num_points, 200)
    test_dataset = AcientDataset('test', opt.dataset_root, opt.num_points, 200)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    criterion = Loss(opt.num_points)
    print('开始训练，共训练{}轮-----------'.format(opt.nepoch))
    my_logger.info('开始训练，共训练{}轮-----------'.format(opt.nepoch))
    train_loss = []
    test_loss = []
    for epoch in range(start_epoch, opt.nepoch):
        model = model.train()
        losses = 0
        for i, data in enumerate(dataloader):
            normal_points,center0,scale, target_mode,target_s,target_num = data

            normal_points, target_mode ,target_s= Variable(normal_points).cuda(), Variable(target_mode).cuda(),\
            Variable(target_s).cuda()
            center0, scale = center0.to(device), scale.to(device)
            optimizer.zero_grad()
            #对称平面上1点，对称法向量3点，足点3点，旋转1点，？ ，对称模式
            pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode = model(normal_points)
            #pred_cent = pred_cent * scale + center0
            postion_points = normal_points[:,:,:3]
            loss, dis, error_cent, loss_ref, error_ref, error_num, error_mode = criterion(
                pred_cent, pred_ref, pred_foot_ref, pred_rot,
                pred_num, pred_mode, target_s, postion_points, opt.w, target_mode)

            loss.backward()
            optimizer.step()
            losses += loss.item()
        t_loss = losses / len(dataloader)
        train_loss.append(t_loss)

        print('第{}轮，训练损失为：{}----------------'.format(epoch,t_loss))
        if epoch % 5 == 0:
            savepath = str(savel_model_path) + '/model.pth'
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        losses_test = 0
        model = model.eval()
        with torch.no_grad():
            for j, data_test in enumerate(dataloader_test):
                normal_points, center0, scale, target_mode, target_s, target_num = data_test
                normal_points, target_mode, target_s = Variable(normal_points).cuda(), Variable(target_mode).cuda(), \
                    Variable(target_s).cuda()
                center0, scale = center0.to(device), scale.to(device)
                pred_cent, pred_ref, pred_foot_ref, pred_rot, pred_num, pred_mode = model(normal_points)
                postion_points = normal_points[:, :, :3]
                loss, dis, error_cent, loss_ref, error_ref, error_num, error_mode = criterion(
                    pred_cent, pred_ref, pred_foot_ref, pred_rot,
                    pred_num, pred_mode, target_s, postion_points, opt.w, target_mode)
                losses_test += loss.item()

            tt_loss = losses_test / len(dataloader_test)
            test_loss.append(tt_loss)
            print('第{}轮，验证损失为：{}'.format(epoch, tt_loss))
            my_logger.info('{} {} {}'.format(epoch, t_loss, tt_loss))
        #scheduler.step()
    plot_curve(train_loss,test_loss)
if __name__ == '__main__':
    main()