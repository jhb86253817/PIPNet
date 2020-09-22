# supervised learning

# 300W, resnet18
#python lib/test.py experiments/data_300W/pip_32_16_60_r18_l2_l1_10_1_nb10.py test.txt images_test
# 300W, resnet101
#python lib/test.py experiments/data_300W/pip_32_16_60_r101_l2_l1_10_1_nb10.py test.txt images_test

# COFW, resnet18
#python lib/test.py experiments/COFW/pip_32_16_60_r18_l2_l1_10_1_nb10.py test.txt images_test
# COFW, resnet101
#python lib/test.py experiments/COFW/pip_32_16_60_r101_l2_l1_10_1_nb10.py test.txt images_test

# WFLW, resnet18
#python lib/test.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py test.txt images_test
# WFLW, resnet101
#python lib/test.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py test.txt images_test

# AFLW, resnet18
#python lib/test.py experiments/AFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py test.txt images_test
# AFLW, resnet101
#python lib/test.py experiments/AFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py test.txt images_test

######################################################################################
# GSSL

# 300W + COFW_68 (unlabeled) + WFLW_68 (unlabeled), resnet18, with curriculum
#python lib/test.py experiments/data_300W_COFW_WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_300W.txt images_test_300W
#python lib/test.py experiments/data_300W_COFW_WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_COFW.txt images_test_COFW
#python lib/test.py experiments/data_300W_COFW_WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_WFLW.txt images_test_WFLW

# 300W + CelebA (unlabeled), resnet18, with curriculum
#python lib/test.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_300W.txt images_test_300W
#python lib/test.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_COFW.txt images_test_COFW
#python lib/test.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py test_WFLW.txt images_test_WFLW
