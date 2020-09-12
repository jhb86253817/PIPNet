import torch.utils.data as data
import torch
from PIL import Image, ImageFilter 
import os, cv2
import numpy as np
import random
from scipy.stats import norm
from math import floor

def random_translate(image, target):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        #c = 30 #left/right (i.e. 5/-5)
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        #f = 30 #up/down (i.e. 5/-5)
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target_translate = target.copy()
        target_translate = target_translate.reshape(-1, 2)
        target_translate[:, 0] -= 1.*c/image_width
        target_translate[:, 1] -= 1.*f/image_height
        target_translate = target_translate.flatten()
        target_translate[target_translate < 0] = 0
        target_translate[target_translate > 1] = 1
        return image, target_translate
    else:
        return image, target

def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*5))
    return image

def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        target = target.flatten()
        return image, target
    else:
        return image, target

def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num= int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num*2) + np.array([center_x, center_y]*landmark_num)
        return image, target_rot
    else:
        return image, target

def gen_target_pip(target, meanface_indices, target_map1, target_map2, target_map3, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])
    map_channel1, map_height1, map_width1 = target_map1.shape
    map_channel2, map_height2, map_width2 = target_map2.shape
    map_channel3, map_height3, map_width3 = target_map3.shape
    target = target.reshape(-1, 2)
    assert map_channel1 == target.shape[0]

    for i in range(map_channel1):
        mu_x1 = int(floor(target[i][0] * map_width1))
        mu_y1 = int(floor(target[i][1] * map_height1))
        mu_x1 = max(0, mu_x1)
        mu_y1 = max(0, mu_y1)
        mu_x1 = min(mu_x1, map_width1-1)
        mu_y1 = min(mu_y1, map_height1-1)
        target_map1[i, mu_y1, mu_x1] = 1

        shift_x = target[i][0] * map_width1 - mu_x1
        shift_y = target[i][1] * map_height1 - mu_y1
        target_local_x[i, mu_y1, mu_x1] = shift_x
        target_local_y[i, mu_y1, mu_x1] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width1 - mu_x1
            nb_y = target[meanface_indices[i][j]][1] * map_height1 - mu_y1
            target_nb_x[num_nb*i+j, mu_y1, mu_x1] = nb_x
            target_nb_y[num_nb*i+j, mu_y1, mu_x1] = nb_y

        mu_x2 = int(floor(target[i][0] * map_width2))
        mu_y2 = int(floor(target[i][1] * map_height2))
        mu_x2 = max(0, mu_x2)
        mu_y2 = max(0, mu_y2)
        mu_x2 = min(mu_x2, map_width2-1)
        mu_y2 = min(mu_y2, map_height2-1)
        target_map2[i, mu_y2, mu_x2] = 1

        mu_x3 = int(floor(target[i][0] * map_width3))
        mu_y3 = int(floor(target[i][1] * map_height3))
        mu_x3 = max(0, mu_x3)
        mu_y3 = max(0, mu_y3)
        mu_x3 = min(mu_x3, map_width3-1)
        mu_y3 = min(mu_y3, map_height3-1)
        target_map3[i, mu_y3, mu_x3] = 1

    return target_map1, target_map2, target_map3, target_local_x, target_local_y, target_nb_x, target_nb_y

def gen_target_pip_cls1(target, target_map1):
    map_channel1, map_height1, map_width1 = target_map1.shape
    target = target.reshape(-1, 2)
    assert map_channel1 == target.shape[0]

    for i in range(map_channel1):
        mu_x1 = int(floor(target[i][0] * map_width1))
        mu_y1 = int(floor(target[i][1] * map_height1))
        mu_x1 = max(0, mu_x1)
        mu_y1 = max(0, mu_y1)
        mu_x1 = min(mu_x1, map_width1-1)
        mu_y1 = min(mu_y1, map_height1-1)
        target_map1[i, mu_y1, mu_x1] = 1

    return target_map1 

def gen_target_pip_cls2(target, target_map2):
    map_channel2, map_height2, map_width2 = target_map2.shape
    target = target.reshape(-1, 2)
    assert map_channel2 == target.shape[0]

    for i in range(map_channel2):
        mu_x2 = int(floor(target[i][0] * map_width2))
        mu_y2 = int(floor(target[i][1] * map_height2))
        mu_x2 = max(0, mu_x2)
        mu_y2 = max(0, mu_y2)
        mu_x2 = min(mu_x2, map_width2-1)
        mu_y2 = min(mu_y2, map_height2-1)
        target_map2[i, mu_y2, mu_x2] = 1

    return target_map2 

def gen_target_pip_cls3(target, target_map3):
    map_channel3, map_height3, map_width3 = target_map3.shape
    target = target.reshape(-1, 2)
    assert map_channel3 == target.shape[0]

    for i in range(map_channel3):
        mu_x3 = int(floor(target[i][0] * map_width3))
        mu_y3 = int(floor(target[i][1] * map_height3))
        mu_x3 = max(0, mu_x3)
        mu_y3 = max(0, mu_y3)
        mu_x3 = min(mu_x3, map_width3-1)
        mu_y3 = min(mu_y3, map_height3-1)
        target_map3[i, mu_y3, mu_x3] = 1

    return target_map3 

class ImageFolder_pip(data.Dataset):
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, meanface_indices, transform=None, target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.meanface_indices = meanface_indices
        self.num_nb = len(meanface_indices[0])
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_name, target_type, target = self.imgs[index]
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        img, target = random_translate(img, target)
        img = random_occlusion(img)
        img, target = random_flip(img, target, self.points_flip)
        img, target = random_rotate(img, target, 30)
        img = random_blur(img)

        target_map1 = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_map2 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/2), int(self.input_size/self.net_stride/2)))
        target_map3 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/4), int(self.input_size/self.net_stride/4)))
        target_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))

        mask_map1 = np.ones((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        mask_map2 = np.ones((self.num_lms, int(self.input_size/self.net_stride/2), int(self.input_size/self.net_stride/2)))
        mask_map3 = np.ones((self.num_lms, int(self.input_size/self.net_stride/4), int(self.input_size/self.net_stride/4)))
        mask_local_x = np.ones((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        mask_local_y = np.ones((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        mask_nb_x = np.ones((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        mask_nb_y = np.ones((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))

        if target_type == 'std':
            target_map1, target_map2, target_map3, target_local_x, target_local_y, target_nb_x, target_nb_y = gen_target_pip(target, self.meanface_indices, target_map1, target_map2, target_map3, target_local_x, target_local_y, target_nb_x, target_nb_y)
            mask_map2 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/2), int(self.input_size/self.net_stride/2)))
            mask_map3 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/4), int(self.input_size/self.net_stride/4)))
        elif target_type == 'cls1':
            target_map1 = gen_target_pip_cls1(target, target_map1)
            mask_map2 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/2), int(self.input_size/self.net_stride/2)))
            mask_map3 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/4), int(self.input_size/self.net_stride/4)))
            mask_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        elif target_type == 'cls2':
            target_map2 = gen_target_pip_cls2(target, target_map2)
            mask_map1 = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_map3 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/4), int(self.input_size/self.net_stride/4)))
            mask_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        elif target_type == 'cls3':
            target_map3 = gen_target_pip_cls3(target, target_map3)
            mask_map1 = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_map2 = np.zeros((self.num_lms, int(self.input_size/self.net_stride/2), int(self.input_size/self.net_stride/2)))
            mask_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            mask_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        else:
            print('No such target type!')
            exit(0)

        target_map1 = torch.from_numpy(target_map1).float()
        target_map2 = torch.from_numpy(target_map2).float()
        target_map3 = torch.from_numpy(target_map3).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()
        mask_map1 = torch.from_numpy(mask_map1).float()
        mask_map2 = torch.from_numpy(mask_map2).float()
        mask_map3 = torch.from_numpy(mask_map3).float()
        mask_local_x = torch.from_numpy(mask_local_x).float()
        mask_local_y = torch.from_numpy(mask_local_y).float()
        mask_nb_x = torch.from_numpy(mask_nb_x).float()
        mask_nb_y = torch.from_numpy(mask_nb_y).float()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_map1 = self.target_transform(target_map1)
            target_map2 = self.target_transform(target_map2)
            target_map3 = self.target_transform(target_map3)
            target_local_x = self.target_transform(target_local_x)
            target_local_y = self.target_transform(target_local_y)
            target_nb_x = self.target_transform(target_nb_x)
            target_nb_y = self.target_transform(target_nb_y)

        return img, target_map1, target_map2, target_map3, target_local_x, target_local_y, target_nb_x, target_nb_y, mask_map1, mask_map2, mask_map3, mask_local_x, mask_local_y, mask_nb_x, mask_nb_y

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    pass
    
