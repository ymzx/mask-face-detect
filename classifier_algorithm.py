from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm
import matplotlib.pyplot as plt, os
import time, logging
from configs.config import classifier_model_param_path

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def precision_recall_accuracy_f1(y, y_pre, classes=1):
    '''
    :param y:
    :param y_pre:
    :param classes: 默认对类别1进行性能评估
    :return:
    '''
    # sklearn直接求解precision, recall, fscore, support
    sk_precision, sk_recall, sk_fscore, sk_support = metrics.precision_recall_fscore_support(y, y_pre)
    precision, recall, f1_score = sk_precision[classes], sk_recall[classes], sk_fscore[classes]
    accuracy = metrics.accuracy_score(y, y_pre)
    '''
    # 手动求类别classes性能指标
    # 求TP、FP、TN、FN
    matrix = metrics.confusion_matrix(y, y_pre)
    TN, FP, FN, TP = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    precision = TP / (TP + FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    f1_score = 2*precision*recall/(precision+recall)
    '''
    return precision, recall, accuracy, f1_score


def draw_roc_curve(y, y_pro, algorithm, auc):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pro)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.plot(fpr, tpr, linewidth=2.0, label=algorithm+', '+'AUC='+str(round(auc,3)))
    plt.legend(loc='lower right')


def importances_bar(model):
    print(model.feature_importances_)
    # feature_important = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    # plt.bar(feature_important.index, feature_important.data)
    # plt.show()


class Classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def gbdt(self):
        # gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5)
        gbc = GradientBoostingClassifier()
        gbc.fit(self.x_train, self.y_train.ravel())
        model_save_path = os.path.join(classifier_model_param_path, 'gbdt_model_paras.m')
        joblib.dump(gbc, model_save_path)   # 保存模型
        y_test_pre = gbc.predict(self.x_test)
        y_test_pro = gbc.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:,1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def svm(self):
        svc = SVC(probability=True, gamma='auto', random_state=10, tol=1e-4)
        svc.fit(self.x_train, self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'svm_model_paras.m')
        joblib.dump(svc, model_save_path)
        y_test_pre = svc.predict(self.x_test)
        y_test_pro = svc.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def rf(self):
        rfc = RandomForestClassifier(n_estimators=200, random_state=10)
        rfc.fit(self.x_train, self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'rf_model_paras.m')
        joblib.dump(rfc, model_save_path)
        y_test_pre = rfc.predict(self.x_test)
        y_test_pro = rfc.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def lr(self):
        lr = LogisticRegression(random_state=10, tol=1e-4, solver='lbfgs', max_iter=1000)
        lr.fit(self.x_train, self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'lr_model_paras.m')
        joblib.dump(lr, model_save_path)
        y_test_pre = lr.predict(self.x_test)
        y_test_pro = lr.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def xgboost(self):
        xgbc = XGBClassifier(random_state=10, use_label_encoder=False)
        xgbc.fit(self.x_train, self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'xgb_model_paras.m')
        joblib.dump(xgbc, model_save_path)
        y_test_pre = xgbc.predict(self.x_test)
        y_test_pro = xgbc.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def adaboost(self):
        adabc = AdaBoostClassifier(random_state=10)
        adabc.fit(self.x_train, self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'adab_model_paras.m')
        joblib.dump(adabc, model_save_path)
        y_test_pre = adabc.predict(self.x_test)
        y_test_pro = adabc.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def lgbm(self):
        lgbm=lightgbm.LGBMClassifier(random_state=10)
        lgbm.fit(self.x_train,self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'lgbm_model_paras.m')
        joblib.dump(lgbm, model_save_path)
        y_test_pre = lgbm.predict(self.x_test)
        y_test_pro = lgbm.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro

    def nb(self):
        nb = MultinomialNB()
        nb.fit(self.x_train,self.y_train)
        model_save_path = os.path.join(classifier_model_param_path, 'nb_model_paras.m')
        joblib.dump(nb, model_save_path)
        y_test_pre = nb.predict(self.x_test)
        y_test_pro = nb.predict_proba(self.x_test)
        precision, recall, accuracy, f1_score = precision_recall_accuracy_f1(self.y_test.ravel(), y_test_pre)
        auc = metrics.roc_auc_score(self.y_test.ravel(), y_test_pro[:, 1])
        return precision, recall, accuracy, f1_score, auc, y_test_pro


def classifier_hub(data, algorithm):
    x_train, x_test, y_train, y_test = data
    classifier = Classifier(x_train, x_test, y_train, y_test)
    precision, recall, accuracy, f1_score, auc = 0, 0, 0, 0, 0
    if not os.path.exists(classifier_model_param_path):
        os.makedirs(classifier_model_param_path)
    if algorithm == 'GBDT':
        # GBDT 梯度提升决策树
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.gbdt()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2-t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'SVM':
        # SVM 支持向量机
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.svm()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'RF':
        # RF 随机森林
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.rf()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'LR':
        # LR 逻辑回归
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.lr()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'XGBoost':
        # XGBoost 极端梯度提升
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.xgboost()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'LGBM':
        # LightGBM
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.lgbm()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'AdaBoost':
        # AdaBoost
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.adaboost()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    elif algorithm == 'NB':
        # Naive Bayes
        logging.info('执行%s' % algorithm)
        t1 = time.time()
        precision, recall, accuracy, f1_score, auc, y_test_pro = classifier.nb()
        t2 = time.time()
        logging.info('%s耗时%s' % (algorithm, round(t2 - t1, 3)))
        # 画ROC曲线
        draw_roc_curve(y_test, y_test_pro[:, 1], algorithm, auc)
    return round(precision, 3), round(recall, 3), round(accuracy, 3), round(f1_score, 3), round(auc, 3)


if __name__ == '__main__':
    # 可选算法
    algorithm_list = ['LR', 'AdaBoost', 'LGBM', 'RF', 'XGBoost', 'SVM']
    print("分类器", '准确率', '召回率', '正确率', 'F1', 'AUC')
    data = None
    for mode in algorithm_list:
        precision, recall, accuracy, f1_score, auc = classifier_hub(data, mode)
        print(mode, precision, recall, accuracy, f1_score, auc)
    plt.show()
