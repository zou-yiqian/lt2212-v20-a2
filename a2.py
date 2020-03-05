import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
from nltk.tokenize import word_tokenize
import re
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

random.seed(42)


# PART 1
# DONT CHANGE THIS FUNCTION
def part1(samples):
    # extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X


def extract_features(samples):
    print("Extracting features ...")
    counts = {}
    for m1 in range(0, len(samples)):
        text = samples[m1]
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text)
        words = [t for t in words if t.isalpha()]
        for word in words:
            if word not in counts:
                counts[word] = {m1: 1}
            else:
                if m1 not in counts[word]:
                    counts[word][m1] = 1
                else:
                    counts[word][m1] += 1

    df = pd.DataFrame(counts).fillna(0)
    df = df.values
    return df


# PART 2
# DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    # Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X, n=10):
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    X_dr = svd.transform(X)
    return X_dr


# PART 3
# DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = KNeighborsClassifier(n_neighbors=5)
    elif clf_id == 2:
        clf = SVC(kernel='linear')
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf


# DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    # PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    # get the model
    clf = get_classifier(clf_id)

    # printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()

    # train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)

    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X, y):
    array = np.zeros((X.shape[0], X.shape[1]+1))
    array[:, :-1] = X
    array[:, -1] = y
    np.random.shuffle(array)
    X = array[:, :-1]
    y = array[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X_train, y_train):
    assert is_classifier(clf)
    clf.fit(X_train, y_train)


def evalute_classifier(clf, X_test, y_test):
    assert is_classifier(clf)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, range(0, 20), list(load_data()[2])))


######
# DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


# DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()

    # PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    # part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    # part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(
            model_id=args.model_id,
            n_dim=args.number_dim_reduce
            )
    
