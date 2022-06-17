from gen_hog_feature_label import hog_feature
from sklearn.model_selection import train_test_split
from classifier_algorithm import classifier_hub, plt


def train_main():
    feature, label = hog_feature()
    data = train_test_split(feature, label, test_size=0.3, random_state=0)
    algorithm_list = ['GBDT', 'LR', 'AdaBoost', 'LGBM', 'RF', 'XGBoost', 'SVM', 'NB']
    print("分类器", '准确率', '召回率', '正确率', 'F1', 'AUC')
    for mode in algorithm_list:
        precision, recall, accuracy, f1_score, auc = classifier_hub(data, mode)
        print(mode, precision, recall, accuracy, f1_score, auc)
    plt.show()


if __name__ == '__main__':
    train_main()
