######################################################################################
# supervised learning

# 300W, resnet18
#python lib/train.py experiments/data_300W/pip_32_16_60_r18_l2_l1_10_1_nb10.py
# 300W, resnet101
#python lib/train.py experiments/data_300W/pip_32_16_60_r101_l2_l1_10_1_nb10.py

# COFW, resnet18
#python lib/train.py experiments/COFW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
# COFW, resnet101
#python lib/train.py experiments/COFW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

# WFLW, resnet18
#python lib/train.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
# WFLW, resnet101
#python lib/train.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

# AFLW, resnet18
#python lib/train.py experiments/AFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py
# AFLW, resnet101
#python lib/train.py experiments/AFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py

######################################################################################
# GSSL

# 300W + COFW_68 (unlabeled) + WFLW_68 (unlabeled), resnet18, with curriculum
#python lib/train_gssl.py experiments/data_300W_COFW_WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py 

# 300W + CelebA (unlabeled), resnet18, with curriculum
#nohup python lib/train_gssl.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py &


