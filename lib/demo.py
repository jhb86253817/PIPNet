import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
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

if not len(sys.argv) == 3:
    print('Format:')
    print('python lib/demo.py config_file image_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
image_file = sys.argv[2]

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

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

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

def demo_image(image_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.6
    det_box_scale = 1.2

    net.eval()
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    detections, _ = detector.detect(image, my_thresh, 1)
    for i in range(len(detections)):
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width-1)
        det_ymax = min(det_ymax, image_height-1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1
        cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (input_size, input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        for i in range(cfg.num_lms):
            x_pred = lms_pred_merge[i*2] * det_width
            y_pred = lms_pred_merge[i*2+1] * det_height
            cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)
    #cv2.imwrite('images/1_out.jpg', image)
    cv2.imshow('1', image)
    cv2.waitKey(0)
        

demo_image(image_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)
