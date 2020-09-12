import os, cv2
import hdf5storage
import numpy as np
import sys

def process_300w(root_folder, folder_name, image_name, label_name, target_size):
    image_path = os.path.join(root_folder, folder_name, image_name)
    label_path = os.path.join(root_folder, folder_name, label_name)

    with open(label_path, 'r') as ff:
        anno = ff.readlines()[3:-1]
        anno = [x.strip().split() for x in anno]
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]
        anno_y = [x[1] for x in anno]
        bbox_xmin = min(anno_x)
        bbox_ymin = min(anno_y)
        bbox_xmax = max(anno_x)
        bbox_ymax = max(anno_y)
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        scale = 1.3
        bbox_xmin -= int((scale-1)/2*bbox_width)
        bbox_ymin -= int((scale-1)/2*bbox_height)
        bbox_width *= scale
        bbox_height *= scale
        bbox_width = int(bbox_width)
        bbox_height = int(bbox_height)
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_width = min(bbox_width, image_width-bbox_xmin-1)
        bbox_height = min(bbox_height, image_height-bbox_ymin-1)
        anno = [[(x-bbox_xmin)/bbox_width, (y-bbox_ymin)/bbox_height] for x,y in anno]

        bbox_xmax = bbox_xmin + bbox_width
        bbox_ymax = bbox_ymin + bbox_height
        image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]
        image_crop = cv2.resize(image_crop, (target_size, target_size))
        return image_crop, anno
        
def process_wflw(anno, target_size):
    image_name = anno[-1]
    image_path = os.path.join('..', 'data', 'WFLW', 'WFLW_images', image_name)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    lms = anno[:196]
    lms = [float(x) for x in lms]
    lms_x = lms[0::2]
    lms_y = lms[1::2]
    lms_x = [x if x >=0 else 0 for x in lms_x] 
    lms_x = [x if x <=image_width else image_width for x in lms_x] 
    lms_y = [y if y >=0 else 0 for y in lms_y] 
    lms_y = [y if y <=image_height else image_height for y in lms_y] 
    lms = [[x,y] for x,y in zip(lms_x, lms_y)]
    lms = [x for z in lms for x in z]
    bbox = anno[196:200]
    bbox = [float(x) for x in bbox]
    attrs = anno[200:206]
    attrs = np.array([int(x) for x in attrs])
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    scale = 1.2
    bbox_xmin -= width * (scale-1)/2
    # remove a part of top area for alignment, see details in paper
    bbox_ymin += height * (scale-1)/2
    bbox_xmax += width * (scale-1)/2
    bbox_ymax += height * (scale-1)/2
    bbox_xmin = max(bbox_xmin, 0)
    bbox_ymin = max(bbox_ymin, 0)
    bbox_xmax = min(bbox_xmax, image_width-1)
    bbox_ymax = min(bbox_ymax, image_height-1)
    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    image_crop = image[int(bbox_ymin):int(bbox_ymax), int(bbox_xmin):int(bbox_xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))

    tmp1 = [bbox_xmin, bbox_ymin]*98
    tmp1 = np.array(tmp1)
    tmp2 = [width, height]*98
    tmp2 = np.array(tmp2)
    lms = np.array(lms) - tmp1
    lms = lms / tmp2
    lms = lms.tolist()
    lms = zip(lms[0::2], lms[1::2])
    return image_crop, list(lms) 

def process_celeba(root_folder, image_name, bbox, target_size):
    image = cv2.imread(os.path.join(root_folder, 'CELEBA', 'img_celeba', image_name))
    image_height, image_width, _ = image.shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    scale = 1.2
    xmin -= width * (scale-1)/2
    # remove a part of top area for alignment, see details in paper
    ymin += height * (scale+0.1-1)/2
    xmax += width * (scale-1)/2
    ymax += height * (scale-1)/2
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_width-1)
    ymax = min(ymax, image_height-1)
    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))
    return image_crop

def process_cofw_68_train(image, bbox, anno, target_size):
    image_height, image_width, _ = image.shape
    anno_x = anno[:29]
    anno_y = anno[29:58]
    xmin, ymin, width, height = bbox
    xmax = xmin + width -1
    ymax = ymin + height -1
    scale = 1.3
    xmin -= width * (scale-1)/2
    ymin -= height * (scale-1)/2
    xmax += width * (scale-1)/2
    ymax += height * (scale-1)/2
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_width-1)
    ymax = min(ymax, image_height-1)
    anno_x = (anno_x - xmin) / (xmax - xmin)
    anno_y = (anno_y - ymin) / (ymax - ymin)
    anno = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1)
    anno = list(anno)
    anno = [list(x) for x in anno]
    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))
    return image_crop, anno

def process_cofw_68_test(image, bbox, anno, target_size):
    image_height, image_width, _ = image.shape
    anno_x = anno[:,0].flatten()
    anno_y = anno[:,1].flatten()

    xmin, ymin, width, height = bbox
    xmax = xmin + width -1
    ymax = ymin + height -1

    scale = 1.3
    xmin -= width * (scale-1)/2
    ymin -= height * (scale-1)/2
    xmax += width * (scale-1)/2
    ymax += height * (scale-1)/2
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_width-1)
    ymax = min(ymax, image_height-1)
    anno_x = (anno_x - xmin) / (xmax - xmin)
    anno_y = (anno_y - ymin) / (ymax - ymin)
    anno = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1)
    anno = list(anno)
    anno = [list(x) for x in anno]
    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))
    return image_crop, anno

def gen_meanface(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'train_300W.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]
    annos = [[float(x) for x in anno] for anno in annos]
    annos = np.array(annos)
    meanface = np.mean(annos, axis=0)
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]
    
    with open(os.path.join(root_folder, data_name, 'meanface.txt'), 'w') as f:
        f.write(' '.join(meanface))

def convert_wflw(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'test_WFLW_98.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split() for x in annos]
    annos_new = []
    for anno in annos:
        annos_new.append([])
        # name
        annos_new[-1].append(anno[0])
        anno = anno[1:]
        # jaw
        for i in range(17):
            annos_new[-1].append(anno[i*2*2])
            annos_new[-1].append(anno[i*2*2+1])
        # left eyebrow
        annos_new[-1].append(anno[33*2])
        annos_new[-1].append(anno[33*2+1])
        annos_new[-1].append(anno[34*2])
        annos_new[-1].append(str((float(anno[34*2+1])+float(anno[41*2+1]))/2))
        annos_new[-1].append(anno[35*2])
        annos_new[-1].append(str((float(anno[35*2+1])+float(anno[40*2+1]))/2))
        annos_new[-1].append(anno[36*2])
        annos_new[-1].append(str((float(anno[36*2+1])+float(anno[39*2+1]))/2))
        annos_new[-1].append(anno[37*2])
        annos_new[-1].append(str((float(anno[37*2+1])+float(anno[38*2+1]))/2))
        # right eyebrow
        annos_new[-1].append(anno[42*2])
        annos_new[-1].append(str((float(anno[42*2+1])+float(anno[50*2+1]))/2))
        annos_new[-1].append(anno[43*2])
        annos_new[-1].append(str((float(anno[43*2+1])+float(anno[49*2+1]))/2))
        annos_new[-1].append(anno[44*2])
        annos_new[-1].append(str((float(anno[44*2+1])+float(anno[48*2+1]))/2))
        annos_new[-1].append(anno[45*2])
        annos_new[-1].append(str((float(anno[45*2+1])+float(anno[47*2+1]))/2))
        annos_new[-1].append(anno[46*2])
        annos_new[-1].append(anno[46*2+1])
        # nose
        for i in range(51, 60):
            annos_new[-1].append(anno[i*2])
            annos_new[-1].append(anno[i*2+1])
        # left eye
        annos_new[-1].append(anno[60*2])
        annos_new[-1].append(anno[60*2+1])
        annos_new[-1].append(str(0.666*float(anno[61*2])+0.333*float(anno[62*2])))
        annos_new[-1].append(str(0.666*float(anno[61*2+1])+0.333*float(anno[62*2+1])))
        annos_new[-1].append(str(0.666*float(anno[63*2])+0.333*float(anno[62*2])))
        annos_new[-1].append(str(0.666*float(anno[63*2+1])+0.333*float(anno[62*2+1])))
        annos_new[-1].append(anno[64*2])
        annos_new[-1].append(anno[64*2+1])
        annos_new[-1].append(str(0.666*float(anno[65*2])+0.333*float(anno[66*2])))
        annos_new[-1].append(str(0.666*float(anno[65*2+1])+0.333*float(anno[66*2+1])))
        annos_new[-1].append(str(0.666*float(anno[67*2])+0.333*float(anno[66*2])))
        annos_new[-1].append(str(0.666*float(anno[67*2+1])+0.333*float(anno[66*2+1])))
        # right eye
        annos_new[-1].append(anno[68*2])
        annos_new[-1].append(anno[68*2+1])
        annos_new[-1].append(str(0.666*float(anno[69*2])+0.333*float(anno[70*2])))
        annos_new[-1].append(str(0.666*float(anno[69*2+1])+0.333*float(anno[70*2+1])))
        annos_new[-1].append(str(0.666*float(anno[71*2])+0.333*float(anno[70*2])))
        annos_new[-1].append(str(0.666*float(anno[71*2+1])+0.333*float(anno[70*2+1])))
        annos_new[-1].append(anno[72*2])
        annos_new[-1].append(anno[72*2+1])
        annos_new[-1].append(str(0.666*float(anno[73*2])+0.333*float(anno[74*2])))
        annos_new[-1].append(str(0.666*float(anno[73*2+1])+0.333*float(anno[74*2+1])))
        annos_new[-1].append(str(0.666*float(anno[75*2])+0.333*float(anno[74*2])))
        annos_new[-1].append(str(0.666*float(anno[75*2+1])+0.333*float(anno[74*2+1])))
        # mouth
        for i in range(76, 96):
            annos_new[-1].append(anno[i*2])
            annos_new[-1].append(anno[i*2+1])

    with open(os.path.join(root_folder, data_name, 'test_WFLW.txt'), 'w') as f:
        for anno in annos_new:
            f.write(' '.join(anno)+'\n')

def gen_data(root_folder, data_name, target_size):
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))
    ################################################################################################################
    if data_name == 'CELEBA':
        os.system('rmdir ../data/CELEBA/images_test')
        with open(os.path.join(root_folder, data_name, 'celeba_bboxes.txt'), 'r') as f:
            bboxes = f.readlines()

        bboxes = [x.strip().split() for x in bboxes]
        with open(os.path.join(root_folder, data_name, 'train.txt'), 'w') as f:
            for bbox in bboxes:
                image_name = bbox[0]
                print(image_name)
                f.write(image_name+'\n')
                bbox = bbox[1:]
                bbox = [int(x) for x in bbox]
                image_crop = process_celeba(root_folder, image_name, bbox, target_size)
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_name), image_crop)
    ################################################################################################################
    elif data_name == 'data_300W_CELEBA':
        os.system('cp -r ../data/CELEBA/images_train ../data/data_300W_CELEBA/.')
        os.system('cp ../data/CELEBA/train.txt ../data/data_300W_CELEBA/train_CELEBA.txt')

        os.system('rmdir ../data/data_300W_CELEBA/images_test')
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_300W')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_300W'))
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_COFW')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_COFW'))
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_WFLW')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_WFLW'))

        # train for data_300W
        folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, 'data_300W', folder_train)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_train, image_name, label_name, target_size)
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
                annos_train[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'train_300W.txt'), 'w') as f:
            for image_crop_name, anno in annos_train.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # test for data_300W
        folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, 'data_300W', folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_test, image_name, label_name, target_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test_300W', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test_300W.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # test for COFW_68
        test_mat = hdf5storage.loadmat(os.path.join('../data/COFW', 'COFW_test_color.mat'))
        images = test_mat['IsT']
        
        bboxes_mat = hdf5storage.loadmat(os.path.join('../data/data_300W_CELEBA', 'cofw68_test_bboxes.mat'))
        bboxes = bboxes_mat['bboxes']
        image_num = images.shape[0]
        with open('../data/data_300W_CELEBA/test_COFW.txt', 'w') as f:
            for i in range(image_num):
                image = images[i,0]
                # grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # swap rgb channel to bgr
                else:
                    image = image[:,:,::-1]
            
                bbox = bboxes[i,:]
                anno_mat = hdf5storage.loadmat(os.path.join('../data/data_300W_CELEBA/cofw68_test_annotations', str(i+1)+'_points.mat'))
                anno = anno_mat['Points']
                image_crop, anno = process_cofw_68_test(image, bbox, anno, target_size)
                pad_num = 4-len(str(i+1))
                image_crop_name = 'cofw_test_' + '0' * pad_num + str(i+1) + '.jpg'
                cv2.imwrite(os.path.join('../data/data_300W_CELEBA/images_test_COFW', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # test for WFLW_68
        test_file = 'list_98pt_rect_attr_test.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', test_file), 'r') as f:
            annos_test = f.readlines()
        annos_test = [x.strip().split() for x in annos_test]
        names_mapping = {}
        count = 1
        with open(os.path.join(root_folder, 'data_300W_CELEBA', 'test_WFLW_98.txt'), 'w') as f:
            for anno_test in annos_test:
                image_crop, anno = process_wflw(anno_test, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_test_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                names_mapping[anno_test[0]+'_'+anno_test[-1]] = [image_crop_name, anno]
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test_WFLW', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in list(anno):
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1

        convert_wflw(root_folder, data_name)

        gen_meanface(root_folder, data_name)
    ################################################################################################################
    elif data_name == 'data_300W_COFW_WFLW':

        os.system('rmdir ../data/data_300W_COFW_WFLW/images_test')
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_300W')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_300W'))
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_COFW')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_COFW'))
        if not os.path.exists(os.path.join(root_folder, data_name, 'images_test_WFLW')):
            os.mkdir(os.path.join(root_folder, data_name, 'images_test_WFLW'))

        # train for data_300W
        folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, 'data_300W', folder_train)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_train, image_name, label_name, target_size)
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
                annos_train[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'train_300W.txt'), 'w') as f:
            for image_crop_name, anno in annos_train.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # test for data_300W
        folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, 'data_300W', folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_test, image_name, label_name, target_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test_300W', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test_300W.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # train for COFW_68
        ###################
        train_file = 'COFW_train_color.mat'
        train_mat = hdf5storage.loadmat(os.path.join(root_folder, 'COFW', train_file))
        images = train_mat['IsTr']
        bboxes = train_mat['bboxesTr']
        annos = train_mat['phisTr']

        count = 1
        with open('../data/data_300W_COFW_WFLW/train_COFW.txt', 'w') as f:
            for i in range(images.shape[0]):
                image = images[i, 0]
                # grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # swap rgb channel to bgr
                else:
                    image = image[:,:,::-1]
                bbox = bboxes[i, :]
                anno = annos[i, :]
                image_crop, anno = process_cofw_68_train(image, bbox, anno, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'cofw_train_' + '0' * pad_num + str(count) + '.jpg'
                f.write(image_crop_name+'\n')
                cv2.imwrite(os.path.join(root_folder, 'data_300W_COFW_WFLW', 'images_train', image_crop_name), image_crop)
                count += 1
        ###################

        # test for COFW_68
        test_mat = hdf5storage.loadmat(os.path.join('../data/COFW', 'COFW_test_color.mat'))
        images = test_mat['IsT']
        
        bboxes_mat = hdf5storage.loadmat(os.path.join('../data/data_300W_COFW_WFLW', 'cofw68_test_bboxes.mat'))
        bboxes = bboxes_mat['bboxes']
        image_num = images.shape[0]
        with open('../data/data_300W_COFW_WFLW/test_COFW.txt', 'w') as f:
            for i in range(image_num):
                image = images[i,0]
                # grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # swap rgb channel to bgr
                else:
                    image = image[:,:,::-1]
            
                bbox = bboxes[i,:]
                anno_mat = hdf5storage.loadmat(os.path.join('../data/data_300W_COFW_WFLW/cofw68_test_annotations', str(i+1)+'_points.mat'))
                anno = anno_mat['Points']
                image_crop, anno = process_cofw_68_test(image, bbox, anno, target_size)
                pad_num = 4-len(str(i+1))
                image_crop_name = 'cofw_test_' + '0' * pad_num + str(i+1) + '.jpg'
                cv2.imwrite(os.path.join('../data/data_300W_COFW_WFLW/images_test_COFW', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        # train for WFLW_68
        train_file = 'list_98pt_rect_attr_train.txt'
        with open(os.path.join('../data', 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', train_file), 'r') as f:
            annos_train = f.readlines()
        annos_train = [x.strip().split() for x in annos_train]
        count = 1
        with open('../data/data_300W_COFW_WFLW/train_WFLW.txt', 'w') as f:
            for anno_train in annos_train:
                image_crop, anno = process_wflw(anno_train, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_train_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                f.write(image_crop_name+'\n')
                cv2.imwrite(os.path.join(root_folder, 'data_300W_COFW_WFLW', 'images_train', image_crop_name), image_crop)
                count += 1

        # test for WFLW_68
        test_file = 'list_98pt_rect_attr_test.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', test_file), 'r') as f:
            annos_test = f.readlines()
        annos_test = [x.strip().split() for x in annos_test]
        names_mapping = {}
        count = 1
        with open(os.path.join(root_folder, 'data_300W_COFW_WFLW', 'test_WFLW_98.txt'), 'w') as f:
            for anno_test in annos_test:
                image_crop, anno = process_wflw(anno_test, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_test_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                names_mapping[anno_test[0]+'_'+anno_test[-1]] = [image_crop_name, anno]
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test_WFLW', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in list(anno):
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1

        convert_wflw(root_folder, data_name)

        gen_meanface(root_folder, data_name)
    else:
        print('Wrong data!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input the data name.')
        print('1. CELEBA')
        print('2. data_300W_CELEBA')
        print('3. data_300W_COFW_WFLW')
        exit(0)
    else:
        data_name = sys.argv[1]
        gen_data('../data', data_name, 256)


