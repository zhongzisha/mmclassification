import pdb
import sys,os,glob,cv2
from skimage.feature import hog
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
import pickle

import pdb


if __name__ == '__main__':

    LABEL_DICT = {
        'nontower': 0,
        'normal': 1,
        'jieduan': 2,
        'wanzhe': 3
    }
    data_root = '/media/ubuntu/SSD/ganta_ensemble/vgg16_b32x2_ganta_with_tower_state'

    save_root = '/media/ubuntu/SSD/ganta_ensemble/'
    os.makedirs(save_root, exist_ok=True)

    network = 'vgg16'
    feat_name = 'fc2'  # conv5, fc1, fc2
    cls_method = 'rf'

    with open(os.path.join(data_root, 'train_{}_feat.npz'.format(feat_name)), 'rb') as fp:
        trainX = pickle.load(fp)
    with open(os.path.join(data_root, 'train_gt_labels.npz'.format(feat_name)), 'rb') as fp:
        trainY = pickle.load(fp)
    with open(os.path.join(data_root, 'val_{}_feat.npz'.format(feat_name)), 'rb') as fp:
        valX = pickle.load(fp)
    with open(os.path.join(data_root, 'val_gt_labels.npz'.format(feat_name)), 'rb') as fp:
        valY = pickle.load(fp)

    trainN = trainX.shape[0]
    valN = valX.shape[0]
    trainX = trainX.reshape((trainN, -1))
    valX = valX.reshape((valN, -1))

    print(trainX.shape)
    sys.exit(-1)

    if cls_method == 'adaboost':
        for n_estimators in [100, 200, 300]:
            print('='*80)
            print('n_estimators: ', n_estimators)
            #
            print('Adaboost')
            clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
            clf.fit(trainX, trainY)
            trainPred = clf.predict(trainX)
            valPred = clf.predict(valX)
            valProb = clf.predict_proba(valX)

            print(confusion_matrix(trainY, trainPred))
            print(classification_report(trainY, trainPred, target_names=list(LABEL_DICT.keys())))
            print(confusion_matrix(valY, valPred))
            print(classification_report(valY, valPred, target_names=list(LABEL_DICT.keys())))

            #
            print('SAMME.R')
            bdt_real = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=2), n_estimators=n_estimators, learning_rate=1
            )
            clf.fit(trainX, trainY)
            trainPred = clf.predict(trainX)
            valPred = clf.predict(valX)
            print(confusion_matrix(trainY, trainPred))
            print(classification_report(trainY, trainPred, target_names=list(LABEL_DICT.keys())))
            print(confusion_matrix(valY, valPred))
            print(classification_report(valY, valPred, target_names=list(LABEL_DICT.keys())))

            print('SAMME')
            bdt_discrete = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=2),
                n_estimators=n_estimators,
                learning_rate=1.5,
                algorithm="SAMME",
            )
            clf.fit(trainX, trainY)
            trainPred = clf.predict(trainX)
            valPred = clf.predict(valX)
            print(confusion_matrix(trainY, trainPred))
            print(classification_report(trainY, trainPred, target_names=list(LABEL_DICT.keys())))
            print(confusion_matrix(valY, valPred))
            print(classification_report(valY, valPred, target_names=list(LABEL_DICT.keys())))

    elif cls_method == 'rf':
        for n_estimators in [100, 200, 300]:
            print('='*80)
            print('n_estimators: ', n_estimators)
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=5,
                                         random_state=0)
            clf.fit(trainX, trainY)
            trainPred = clf.predict(trainX)
            valPred = clf.predict(valX)
            print(confusion_matrix(trainY, trainPred))
            print(classification_report(trainY, trainPred, target_names=list(LABEL_DICT.keys())))
            print(confusion_matrix(valY, valPred))
            print(classification_report(valY, valPred, target_names=list(LABEL_DICT.keys())))
