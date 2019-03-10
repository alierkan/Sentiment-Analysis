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

def normalize2(data):
    return (data - min(data)) / (max(data) - min(data))

def load(e, i, classifiers=["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN", "WARV"], path="../resources/imdb/scores"):
    """ returns an array containing the scores of the specified classifiers.
    Note that we center the scores to 0."""
    assert e in [test_label, valid_label]
    probas = []
    for c in classifiers:
        assert c in ["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN", "WARV"]
        try:
            if c in ["RNNLM", "PARAGRAPH", "NBSVM"]:
                #data = np.loadtxt(os.path.join(path,"-".join([c, e])))
                data = pd.read_csv(os.path.join(path,"-".join([c, e])), delimiter=" ", header=None).values
            elif c in ["ARV", "WARV"]:
                #data = np.loadtxt(os.path.join(path, "_".join(["-".join([c, e]), str(i)])))
                data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i)])), delimiter=" ", header=None).values
            else:
                #data = np.loadtxt(os.path.join(path, "_".join(["-".join([c, e]), str(i%5)])))
                data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i%5)])), delimiter=" ", header=None).values
        except Exception as err:
            print("error=" + err)
        if "RNNLM" in c:
            data = normalize2(data[:,2])
        elif "PARAGRAPH" in c or "NBSVM" in c or "ARV" in c or "CNN" in c or "WARV" in c:
            data = normalize2(data[:, 2])
        probas += [data]
    x = np.vstack(probas).T
    y = np.vstack([np.ones((x.shape[0]//2, 1)), np.zeros((x.shape[0]//2, 1))])
    return x, y


def deep(train, test):
    x_train, y_train = train
    x_test, y_test = test
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(x_train.shape[1], activation='relu'))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(x_train.shape[1], activation='relu'))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=25, batch_size=25)
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
    pairs = [["RNNLM", "PARAGRAPH"],
             ["RNNLM", "NBSVM"],
             ["RNNLM", "ARV"],
             ["RNNLM", "CNN"],
             ["PARAGRAPH", "NBSVM"],
             ["PARAGRAPH", "ARV"],
             ["PARAGRAPH", "CNN"],
             ["NBSVM", "ARV"],
             ["NBSVM", "CNN"],
             ["ARV", "CNN"],
             ["WARV", "RNNLM"],
             ["WARV", "PARAGRAPH"],
             ["WARV", "NBSVM"],
             ["WARV", "ARV"],
             ["WARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM"],
             ["RNNLM", "PARAGRAPH", "ARV"],
             ["RNNLM", "PARAGRAPH", "CNN"],
             ["RNNLM", "NBSVM", "ARV"],
             ["RNNLM", "NBSVM", "CNN"],
             ["RNNLM", "ARV", "CNN"],
             ["PARAGRAPH", "NBSVM", "ARV"],
             ["PARAGRAPH", "NBSVM", "CNN"],
             ["PARAGRAPH", "ARV", "CNN"],
             ["NBSVM", "ARV", "CNN"],
             ["WARV", "PARAGRAPH", "NBSVM"],
             ["WARV", "PARAGRAPH", "ARV"],
             ["WARV", "PARAGRAPH", "CNN"],
             ["WARV", "NBSVM", "ARV"],
             ["WARV", "NBSVM", "CNN"],
             ["WARV", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "WARV"],
             ["RNNLM", "WARV", "ARV"],
             ["RNNLM", "WARV", "CNN"],
             ["RNNLM", "NBSVM", "WARV"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "ARV"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "WARV"],
             ["RNNLM", "PARAGRAPH", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "ARV", "WARV"],
             ["PARAGRAPH", "NBSVM", "ARV", "CNN"],
             ["PARAGRAPH", "NBSVM", "ARV", "WARV"],
             ["PARAGRAPH", "NBSVM", "CNN", "WARV"],
             ["PARAGRAPH", "WARV", "ARV", "CNN"],
             ["RNNLM", "NBSVM", "ARV", "CNN"],
             ["RNNLM", "NBSVM", "ARV", "WARV"],
             ["RNNLM", "NBSVM", "CNN", "WARV"],
             ["NBSVM", "WARV", "ARV", "CNN"],
             ["RNNLM", "WARV", "ARV", "CNN"],
             ["PARAGRAPH", "WARV", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN"],
             ["WARV", "PARAGRAPH", "NBSVM", "ARV", "CNN"],
             ["RNNLM", "WARV", "NBSVM", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "WARV", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "WARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "WARV"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN", "WARV"]
            ]

    with open("../resources/imdb/scores/ensemble.txt", 'w') as out:
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