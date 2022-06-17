import os
import shutil

import cv2
import time
import joblib
from configs.config import img_size


class Classifier:
    def __init__(self, model_param_path):
        self.model = joblib.load(model_param_path)

    def run(self, feature):
        labels = self.model.predict(feature).tolist()
        prob = self.model.predict_proba(feature)
        scores = [prob[i][labels[i]] for i, pro in enumerate(prob)]
        return labels, scores

    @staticmethod
    def preprocess(img_path):
        img = cv2.imread(img_path) if type(img_path) is str else img_path
        img_resize = cv2.resize(img, img_size)
        win_size = img_size
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbin = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbin)
        hist = hog.compute(img_resize)
        feature = hist.reshape(1, -1)
        return feature

    def postprocess(self):
        pass


if __name__ == '__main__':
    img_dir = r'E:\hog_svm_classifier\train_data\neg'
    error_dir = r'E:\hog_svm_classifier\train_data\error2'
    model_save_path = r'E:\hog_svm_classifier\ckpt\classifier_model_param\lr_model_paras.m'
    classifier = Classifier(model_save_path)
    imgs_names = os.listdir(img_dir)
    tmp = []
    for img_name in imgs_names:
        img_path = os.path.join(img_dir, img_name)
        t1 = time.time()
        feat = classifier.preprocess(img_path)
        labels, scores = classifier.run(feat)
        if labels[0] == 1:
            shutil.copyfile(src=img_path, dst=os.path.join(error_dir, img_name))
            tmp.append(scores[0])
            print(labels, scores)
        t2 = time.time()
        print('耗时:', round(t2-t1, 6)) # 耗时小于1ms
    print('平均置信度：', sum(tmp)/len(tmp))
    print(tmp)
