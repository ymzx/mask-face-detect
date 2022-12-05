# mask_face_detect
We combine image processing and machine learning techniques, provide a way to tell if a face is wearing a mask.

## find_hard_examples.py
寻找难例，即挑选出模型识别错误的样本，这些样本一般需要数据增强然后重新训练。

## gen_hog_feature_label.py
生成样本的hog特征以及正负样本的label标签

## predict.py
加载训练好的模型，预测推理

## train.py
加载标记样本，位于train_data目录，按照要求下载即可，进行训练

## classifier_algorithm.py
分类算法、包含可供选择的各种分类模型

## configs/
项目配置文件

## ckpt/
训练完成后的模型存放路径
