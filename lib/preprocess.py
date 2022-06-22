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
        scale = 1.1 
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
        
def process_cofw(image, bbox, anno, target_size):
    image_height, image_width, _ = image.shape
    anno_x = anno[:29]
    anno_y = anno[29:58]
    ################################
    xmin, ymin, width, height = bbox
    xmax = xmin + width -1
    ymax = ymin + height -1
    ################################
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
    bbox_ymin -= height * (scale-1)/2
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

def process_aflw(root_folder, image_name, bbox, anno, target_size):
    image = cv2.imread(os.path.join(root_folder, 'AFLW', 'flickr', image_name))
    image_height, image_width, _ = image.shape
    anno_x = anno[:19]
    anno_y = anno[19:]
    anno_x = [x if x >=0 else 0 for x in anno_x] 
    anno_x = [x if x <=image_width else image_width for x in anno_x] 
    anno_y = [y if y >=0 else 0 for y in anno_y] 
    anno_y = [y if y <=image_height else image_height for y in anno_y] 
    anno_x_min = min(anno_x)
    anno_x_max = max(anno_x)
    anno_y_min = min(anno_y)
    anno_y_max = max(anno_y)
    xmin, xmax, ymin, ymax = bbox
    
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_width-1)
    ymax = min(ymax, image_height-1)

    image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
    image_crop = cv2.resize(image_crop, (target_size, target_size))

    anno_x = (np.array(anno_x) - xmin) / (xmax - xmin)
    anno_y = (np.array(anno_y) - ymin) / (ymax - ymin)

    anno = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1).flatten()
    anno = zip(anno[0::2], anno[1::2])
    return image_crop, anno

def gen_meanface(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'train.txt'), 'r') as f:
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
    with open(os.path.join('../data/WFLW/test.txt'), 'r') as f:
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

    with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
        for anno in annos_new:
            f.write(' '.join(anno)+'\n')


def gen_data(root_folder, data_name, target_size):
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))

    ################################################################################################################
    if data_name == 'data_300W':
        folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_train)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_train, image_name, label_name, target_size)
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
                annos_train[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'train.txt'), 'w') as f:
            for image_crop_name, anno in annos_train.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        

        folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_test, image_name, label_name, target_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        annos = None
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'r') as f:
            annos = f.readlines()
        with open(os.path.join(root_folder, data_name, 'test_common.txt'), 'w') as f:
            for anno in annos:
                if not 'ibug' in anno:
                    f.write(anno)
        with open(os.path.join(root_folder, data_name, 'test_challenge.txt'), 'w') as f:
            for anno in annos:
                if 'ibug' in anno:
                    f.write(anno)

        gen_meanface(root_folder, data_name)
    ################################################################################################################
    elif data_name == 'COFW':
        train_file = 'COFW_train_color.mat'
        train_mat = hdf5storage.loadmat(os.path.join(root_folder, 'COFW', train_file))
        images = train_mat['IsTr']
        bboxes = train_mat['bboxesTr']
        annos = train_mat['phisTr']

        count = 1
        with open(os.path.join(root_folder, 'COFW', 'train.txt'), 'w') as f:
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
                image_crop, anno = process_cofw(image, bbox, anno, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'cofw_train_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, 'COFW', 'images_train', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1

        test_file = 'COFW_test_color.mat'
        test_mat = hdf5storage.loadmat(os.path.join(root_folder, 'COFW', test_file))
        images = test_mat['IsT']
        bboxes = test_mat['bboxesT']
        annos = test_mat['phisT']

        count = 1
        with open(os.path.join(root_folder, 'COFW', 'test.txt'), 'w') as f:
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
                image_crop, anno = process_cofw(image, bbox, anno, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'cofw_test_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, 'COFW', 'images_test', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1
        gen_meanface(root_folder, data_name)
    ################################################################################################################
    elif data_name == 'WFLW':
        train_file = 'list_98pt_rect_attr_train.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', train_file), 'r') as f:
            annos_train = f.readlines()
        annos_train = [x.strip().split() for x in annos_train]
        count = 1
        with open(os.path.join(root_folder, 'WFLW', 'train.txt'), 'w') as f:
            for anno_train in annos_train:
                image_crop, anno = process_wflw(anno_train, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_train_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, 'WFLW', 'images_train', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1

        test_file = 'list_98pt_rect_attr_test.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_rect_attr_train_test', test_file), 'r') as f:
            annos_test = f.readlines()
        annos_test = [x.strip().split() for x in annos_test]
        names_mapping = {}
        count = 1
        with open(os.path.join(root_folder, 'WFLW', 'test.txt'), 'w') as f:
            for anno_test in annos_test:
                image_crop, anno = process_wflw(anno_test, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_test_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                names_mapping[anno_test[0]+'_'+anno_test[-1]] = [image_crop_name, anno]
                cv2.imwrite(os.path.join(root_folder, 'WFLW', 'images_test', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in list(anno):
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1

        test_pose_file = 'list_98pt_test_largepose.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_pose_file), 'r') as f:
            annos_pose_test = f.readlines()
        names_pose = [x.strip().split() for x in annos_pose_test]
        names_pose = [x[0]+'_'+x[-1] for x in names_pose]
        with open(os.path.join(root_folder, 'WFLW', 'test_pose.txt'), 'w') as f:
            for name_pose in names_pose:
                if name_pose in names_mapping:
                    image_crop_name, anno = names_mapping[name_pose]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)

        test_expr_file = 'list_98pt_test_expression.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_expr_file), 'r') as f:
            annos_expr_test = f.readlines()
        names_expr = [x.strip().split() for x in annos_expr_test]
        names_expr = [x[0]+'_'+x[-1] for x in names_expr]
        with open(os.path.join(root_folder, 'WFLW', 'test_expr.txt'), 'w') as f:
            for name_expr in names_expr:
                if name_expr in names_mapping:
                    image_crop_name, anno = names_mapping[name_expr]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)

        test_illu_file = 'list_98pt_test_illumination.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_illu_file), 'r') as f:
            annos_illu_test = f.readlines()
        names_illu = [x.strip().split() for x in annos_illu_test]
        names_illu = [x[0]+'_'+x[-1] for x in names_illu]
        with open(os.path.join(root_folder, 'WFLW', 'test_illu.txt'), 'w') as f:
            for name_illu in names_illu:
                if name_illu in names_mapping:
                    image_crop_name, anno = names_mapping[name_illu]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)

        test_mu_file = 'list_98pt_test_makeup.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_mu_file), 'r') as f:
            annos_mu_test = f.readlines()
        names_mu = [x.strip().split() for x in annos_mu_test]
        names_mu = [x[0]+'_'+x[-1] for x in names_mu]
        with open(os.path.join(root_folder, 'WFLW', 'test_mu.txt'), 'w') as f:
            for name_mu in names_mu:
                if name_mu in names_mapping:
                    image_crop_name, anno = names_mapping[name_mu]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)

        test_occu_file = 'list_98pt_test_occlusion.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_occu_file), 'r') as f:
            annos_occu_test = f.readlines()
        names_occu = [x.strip().split() for x in annos_occu_test]
        names_occu = [x[0]+'_'+x[-1] for x in names_occu]
        with open(os.path.join(root_folder, 'WFLW', 'test_occu.txt'), 'w') as f:
            for name_occu in names_occu:
                if name_occu in names_mapping:
                    image_crop_name, anno = names_mapping[name_occu]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)


        test_blur_file = 'list_98pt_test_blur.txt'
        with open(os.path.join(root_folder, 'WFLW', 'WFLW_annotations', 'list_98pt_test', test_blur_file), 'r') as f:
            annos_blur_test = f.readlines()
        names_blur = [x.strip().split() for x in annos_blur_test]
        names_blur = [x[0]+'_'+x[-1] for x in names_blur]
        with open(os.path.join(root_folder, 'WFLW', 'test_blur.txt'), 'w') as f:
            for name_blur in names_blur:
                if name_blur in names_mapping:
                    image_crop_name, anno = names_mapping[name_blur]
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')
                else:
                    print('error!')
                    exit(0)
        gen_meanface(root_folder, data_name)
    ################################################################################################################
    elif data_name == 'AFLW':
        mat = hdf5storage.loadmat('../data/AFLW/AFLWinfo_release.mat')
        bboxes = mat['bbox']
        annos = mat['data']
        mask_new = mat['mask_new']
        nameList = mat['nameList']
        ra = mat['ra'][0]
        train_indices = ra[:20000]
        test_indices = ra[20000:]

        with open(os.path.join(root_folder, 'AFLW', 'train.txt'), 'w') as f:
            for index in train_indices:
                # from matlab index
                image_name = nameList[index-1][0][0]
                bbox = bboxes[index-1]
                anno = annos[index-1]
                image_crop, anno = process_aflw(root_folder, image_name, bbox, anno, target_size)
                pad_num = 5-len(str(index))
                image_crop_name = 'aflw_train_' + '0' * pad_num + str(index) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, 'AFLW', 'images_train', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        with open(os.path.join(root_folder, 'AFLW', 'test.txt'), 'w') as f:
            for index in test_indices:
                # from matlab index
                image_name = nameList[index-1][0][0]
                bbox = bboxes[index-1]
                anno = annos[index-1]
                image_crop, anno = process_aflw(root_folder, image_name, bbox, anno, target_size)
                pad_num = 5-len(str(index))
                image_crop_name = 'aflw_test_' + '0' * pad_num + str(index) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, 'AFLW', 'images_test', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        gen_meanface(root_folder, data_name)
    ################################################################################################################
    elif data_name == 'LaPa':
        pass
        # TODO
    else:
        print('Wrong data!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input the data name.')
        print('1. data_300W')
        print('2. COFW')
        print('3. WFLW')
        print('4. AFLW')
        print('5. LaPa')
        exit(0)
    else:
        data_name = sys.argv[1]
        gen_data('../data', data_name, 256)


