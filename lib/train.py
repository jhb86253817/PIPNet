import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import copy
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import * 
from mobilenetv3 import mobilenetv3_large

if not len(sys.argv) == 2:
    print('Format:')
    print('python lib/train.py config_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
    os.mkdir(os.path.join('./snapshots', cfg.data_name))
save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

print('###########################################')
print('experiment_name:', cfg.experiment_name)
print('data_name:', cfg.data_name)
print('det_head:', cfg.det_head)
print('net_stride:', cfg.net_stride)
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('decay_steps:', cfg.decay_steps)
print('input_size:', cfg.input_size)
print('backbone:', cfg.backbone)
print('pretrained:', cfg.pretrained)
print('criterion_cls:', cfg.criterion_cls)
print('criterion_reg:', cfg.criterion_reg)
print('cls_loss_weight:', cfg.cls_loss_weight)
print('reg_loss_weight:', cfg.reg_loss_weight)
print('num_lms:', cfg.num_lms)
print('save_interval:', cfg.save_interval)
print('num_nb:', cfg.num_nb)
print('use_gpu:', cfg.use_gpu)
print('gpu_id:', cfg.gpu_id)
print('###########################################')
logging.info('###########################################')
logging.info('experiment_name: {}'.format(cfg.experiment_name))
logging.info('data_name: {}'.format(cfg.data_name))
logging.info('det_head: {}'.format(cfg.det_head))
logging.info('net_stride: {}'.format(cfg.net_stride))
logging.info('batch_size: {}'.format(cfg.batch_size))
logging.info('init_lr: {}'.format(cfg.init_lr))
logging.info('num_epochs: {}'.format(cfg.num_epochs))
logging.info('decay_steps: {}'.format(cfg.decay_steps))
logging.info('input_size: {}'.format(cfg.input_size))
logging.info('backbone: {}'.format(cfg.backbone))
logging.info('pretrained: {}'.format(cfg.pretrained))
logging.info('criterion_cls: {}'.format(cfg.criterion_cls))
logging.info('criterion_reg: {}'.format(cfg.criterion_reg))
logging.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
logging.info('reg_loss_weight: {}'.format(cfg.reg_loss_weight))
logging.info('num_lms: {}'.format(cfg.num_lms))
logging.info('save_interval: {}'.format(cfg.save_interval))
logging.info('num_nb: {}'.format(cfg.num_nb))
logging.info('use_gpu: {}'.format(cfg.use_gpu))
logging.info('gpu_id: {}'.format(cfg.gpu_id))
logging.info('###########################################')

if cfg.det_head == 'pip':
    meanface_indices, _, _, _ = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)


if cfg.det_head == 'pip':
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=cfg.pretrained)
        net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=cfg.pretrained)
        net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v2':
        mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
        net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v3':
        mbnet = mobilenetv3_large()
        if cfg.pretrained:
            mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)
else:
    print('No such head:', cfg.det_head)
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

criterion_cls = None
if cfg.criterion_cls == 'l2':
    criterion_cls = nn.MSELoss()
elif cfg.criterion_cls == 'l1':
    criterion_cls = nn.L1Loss()
else:
    print('No such cls criterion:', cfg.criterion_cls)

criterion_reg = None
if cfg.criterion_reg == 'l1':
    criterion_reg = nn.L1Loss()
elif cfg.criterion_reg == 'l2':
    criterion_reg = nn.MSELoss()
else:
    print('No such reg criterion:', cfg.criterion_reg)

points_flip = None
if cfg.data_name == 'data_300W':
    points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 68
elif cfg.data_name == 'WFLW':
    points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
    assert len(points_flip) == 98
elif cfg.data_name == 'COFW':
    points_flip = [2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 12, 11, 15, 16, 13, 14, 18, 17, 20, 19, 21, 22, 24, 23, 25, 26, 27, 28, 29]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 29
elif cfg.data_name == 'AFLW':
    points_flip = [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 15, 14, 13, 18, 17, 16, 19]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 19
elif cfg.data_name == 'LaPa':
    points_flip = [33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 47, 46, 45, 44, 43, 51, 50, 49, 48, 38, 37, 36, 35, 34, 42, 41, 40, 39, 52, 53, 54, 55, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 80, 79, 78, 77, 76, 83, 82, 81, 84, 71, 70, 69, 68, 67, 74, 73, 72, 75, 91, 90, 89, 88, 87, 86, 85, 96, 95, 94, 93, 92, 101, 100, 99, 98, 97, 104, 103, 102, 106, 105]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 106
else:
    print('No such data!')
    exit(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if cfg.pretrained:  
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
else:
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

labels = get_label(cfg.data_name, 'train.txt')

if cfg.det_head == 'pip':
    train_data = data_utils.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train'), 
                                              labels, cfg.input_size, cfg.num_lms, 
                                              cfg.net_stride, points_flip, meanface_indices,
                                              transforms.Compose([
                                              transforms.RandomGrayscale(0.2),
                                              transforms.ToTensor(),
                                              normalize]))
else:
    print('No such head:', cfg.det_head)
    exit(0)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)

