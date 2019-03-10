import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
from sklearn import metrics
import pandas as pd

test_label = "TEST"
valid_label = "VALID"

def normalize(data):
    norm = np.sqrt((data ** 2).sum())
    return data / norm


def load(e, i, classifiers=["CNN", "ARV", "WARV"], path="../resources/yelp/scores"):
    """ returns an array containing the scores of the specified classifiers.
    Note that we center the scores to 0."""
    assert e in [test_label, valid_label]
    first = True
    for c in classifiers:
        try:
            if c in ["CNN"]:
                data = pd.read_csv(os.path.join(path,"-".join([c, e])), delimiter=" ", header=None).values
            elif c in ["ARV", "WARV"]:
                data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i)])), delimiter=" ", header=None).values
        except Exception as err:
            print("error=" + str(err))
        if first:
            x = normalize(data[:,1:])
            first = False
        else:
            x = np.concatenate((x, normalize(data[:,1:])), axis=1)
    y = np.vstack([np.ones((x.shape[0] // 2, 1)), np.zeros((x.shape[0] // 2, 1))])
    return x, y


def deep(train, test):
    x_train, y_train = train
    x_test, y_test = test
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(x_train.shape[1], activation='relu'))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(x_train.shape[1], activation='relu'))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, batch_size=100)
    result = model.evaluate(x_test, y_test)
    return result


def lr(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l1')
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def svm(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn import svm
    model = svm.SVC()
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def nb(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def knn(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def dt(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def adaboost(train, test):
    x_train, y_train = train
    x_test, y_test = test
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
    model.fit(x_train, y_train.ravel())
    y_pred = model.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    return accuracy(y_pred)


def accuracy(y_pred):
    half = y_pred.shape[0]//2
    ones = y_pred[:half].sum()
    zeros = half - y_pred[half:].sum()
    return 0, (ones+zeros)/(half*2)


if __name__ == "__main__":
    print("\n")
    # duo
    pairs = [["WARV", "CNN"],
             ["WARV", "ARV"],
             ["ARV", "CNN"],
             ["WARV", "ARV", "CNN"]
             ]
    with open("../resources/yelp/scores/ensemble.txt", 'w') as out:
        for c in pairs:
            results = []
            dic = {}
            for i in range(10):
                valid, test = load(valid_label, i, c), load(test_label, i, c)
                l,a = deep(valid, test)
                results.append(tuple((l,a)))
                out.write('-'.join(c) + " = " + str(a))
            dic['-'.join(c)] = results

        for pair, results in dic.items():
            acc = []
            for r in results:
                acc.append(r[1])
                print(pair, "\t lost = {0} / acc = {1}".format(r[0], r[1]))

            np_acc = np.array(acc)
            out.write("Mean = " + str(np_acc.mean()) + "\n")
            out.write("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)) + "\n")
            out.write("\n")