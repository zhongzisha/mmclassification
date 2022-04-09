import pdb
import sys,os,glob,cv2
from skimage.feature import hog
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

import pdb


def extract_features(subset):

    LABEL_DICT = {
        'nontower': 0,
        'normal': 1,
        'jieduan': 2,
        'wanzhe': 3
    }
    data_root = '/media/ubuntu/SSD/ganta_patch_classification/'

    features = []
    labels = []
    for label_name, label in LABEL_DICT.items():
        filenames = glob.glob(os.path.join(data_root, subset, label_name, '*.jpg'))
        labels.append(label * np.ones((len(filenames),), dtype=np.uint8))
        for filename in filenames:
            print(filename)
            image = cv2.imread(filename)
            fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=False, multichannel=True)
            features.append(fd)

    features = np.array(features)
    labels = np.array(labels)

    pdb.set_trace()

    return features, labels


if __name__ == '__main__':
    X_trn, y_trn = extract_features(subset='train')
    X_val, y_val = extract_features(subset='val')

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_trn, y_trn)
    pred_trn = clf.predict(X_trn)
    pred_val = clf.predict(X_val)

    pdb.set_trace()