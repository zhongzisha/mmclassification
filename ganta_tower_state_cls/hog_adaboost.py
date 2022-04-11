import pdb
import sys,os,glob,cv2
from skimage.feature import hog
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle

import pdb

LABEL_DICT = {
    'nontower': 0,
    'normal': 1,
    'jieduan': 2,
    'wanzhe': 3
}
data_root = '/media/ubuntu/SSD/ganta_patch_classification/'


def extract_features(subset):


    features = []
    labels = []
    for label_name, label in LABEL_DICT.items():
        filenames = glob.glob(os.path.join(data_root, subset, label_name, '*.jpg'))
        labels.append(label * np.ones((len(filenames),), dtype=np.uint8))
        for filename in filenames:
            print(filename)
            image = cv2.imread(filename)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, (224, 224))
            fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=False, multichannel=True)
            features.append(fd.reshape((1, -1)))

    # pdb.set_trace()

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # pdb.set_trace()

    return features, labels


if __name__ == '__main__':

    save_root = '/media/ubuntu/SSD/ganta_ensemble/'
    os.makedirs(save_root, exist_ok=True)

    fea_method = 'hog'
    cls_method = 'rf'

    fea_filename = os.path.join(save_root, fea_method + '.pkl')
    if not os.path.exists(fea_filename):
        trainX, trainY = extract_features(subset='train')
        valX, valY = extract_features(subset='val')
        with open(fea_filename, 'wb') as fp:
            pickle.dump({
                'trainX': trainX, 'trainY': trainY,
                'valX': valX, 'valY': valY
            }, fp)
    else:
        with open(fea_filename, 'rb') as fp:
            datadict = pickle.load(fp)
            trainX = datadict['trainX']
            trainY = datadict['trainY']
            valX = datadict['valX']
            valY = datadict['valY']

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
