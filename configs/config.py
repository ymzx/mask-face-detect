import os, sys

base_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # main函数运行父路径
# 分类器模型参数存放路径
classifier_model_param_path = os.path.join(base_dir, r'ckpt\classifier_model_param')
# 样本尺寸
img_size = (64, 80)
# 训练样本和测试样本路径
neg_imgs_path, pos_imgs_path = 'train_data/neg', 'train_data/pos'

# 指定分类器
classifier_list = ['GBDT', 'LR', 'AdaBoost', 'LGBM', 'RF', 'XGBoost', 'SVM', 'NB']


