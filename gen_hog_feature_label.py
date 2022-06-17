import os
import cv2
import numpy as np
from configs.config import pos_imgs_path, neg_imgs_path, img_size


def hog_feature_label(path, num, featureNum, hog, flag):
    featureArray = np.zeros((num, featureNum), np.float32)
    labelArray = np.zeros((num, 1), np.int32)
    image_path_listdir = os.listdir(path)
    for i, image_name in enumerate(image_path_listdir):
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path, 0) # 灰度图
        hist = hog.compute(img)
        for j in range(featureNum):
            featureArray[i, j] = hist[j]
        if flag == '1':
            labelArray[i, 0] = 1 # 正样本标记为1
    return featureArray, labelArray


def hog_feature():
    pos_num = len(os.listdir(pos_imgs_path))  # 正样本数量
    neg_num = len(os.listdir(neg_imgs_path))  # 正样本数量
    print('正样本数量:%s，负样本数量:%s' % (pos_num, neg_num))  # win_size 两个维度均为16倍数
    win_size = img_size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbin = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbin)
    feature_num = int(nbin) * int(block_size[0]/cell_size[0]) * int(block_size[1]/cell_size[1]) * int((win_size[0]-block_size[0])/block_stride[0]+1) * int((win_size[1]-block_size[1])/block_stride[1]+1)
    print('HoG特征数：', feature_num)
    feature_pos, label_pos = hog_feature_label(pos_imgs_path, pos_num, feature_num, hog, '1') # 产生正样本的hog特征以及对应的label
    feature_neg, label_neg = hog_feature_label(neg_imgs_path, neg_num, feature_num, hog, '0')  # 产生负样本的hog特征以及对应的label
    feature = np.vstack((feature_pos, feature_neg))
    label = np.vstack((label_pos, label_neg)).ravel()
    # print(feature_pos.shape, feature_neg.shape, feature.shape, label_pos.shape, label_neg.shape, label.shape)
    print('已产生hog特征！')
    return feature, label


if __name__ == '__main__':
    pass
