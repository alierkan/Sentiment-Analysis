import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
from sklearn import metrics
from keras import  utils
from keras import losses
import pandas as pd

test_label = "TEST"
valid_label = "VALID"
size_test = (611,204,44)
#size_test = (397,160,30)
size_train = (1114,516,78)
#size_train = (1095,549,64)

def normalize(data):
    norm = np.sqrt((data ** 2).sum())
    return data / norm


def load(i, e, classifiers=["CNN", "WARV", "ARV"], path="../resources/semeval/scores"):
    """ returns an array containing the scores of the specified classifiers.
    Note that we center the scores to 0."""
    assert e in [test_label, valid_label]
    first = True
    for c in classifiers:
        try:
            data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i)])), delimiter=" ", header=None).values
        except Exception as err:
            print("error=" + str(err))
        if first:
            x = data[:,1:]
            first = False
        else:
            x = np.concatenate((x, data[:,1:]), axis=1)
    if e == test_label:
        y = np.vstack([np.ones((size_test[0], 1)), np.full((size_test[1], 1), -1), np.zeros((size_test[2], 1))]) #587
    if e == valid_label:
        y = np.vstack([np.ones((size_train[0], 1)), np.full((size_train[1], 1), -1), np.zeros((size_train[2], 1))]) #1708
    return x, y


def deep(train, test):
    num_classes = 3
    x_train, y_train = train
    x_test, y_test = test
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    model = models.Sequential()
    model.add(layers.Dense(x_train.shape[1]//2, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    # model.add(layers.Dense(x_train.shape[1]//2, activation='relu'))
    # model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    # model.add(layers.Dense(x_train.shape[1], activation='relu'))
    # model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adadelta(), loss=losses.binary_crossentropy, metrics=['accuracy'])
    #model.compile(optimizer=optimizers.Adadelta(), loss=losses.categorical_crossentropy,metrics=['categorical_accuracy'])

    history = model.fit(x_train, y_train, epochs=20, batch_size=20)

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
    cnt = 0
    for i in range(0,size_test[0]):
        if y_pred[i] == 1:
            cnt += 1
    for i in range(size_test[0]+1,size_test[0]+size_test[1]):
        if y_pred[i] == -1:
            cnt += 1
    for i in range(size_test[0]+size_test[1]+1,size_test[0]+size_test[1]+size_test[2]):
        if y_pred[i] == 0:
            cnt += 1
    return 0, cnt / (size_test[0]+size_test[1]+size_test[2])


if __name__ == "__main__":
    print("\n")
    # duo
    pairs = [["ARV", "CNN"],
             ["ARV", "WARV"],
             ["WARV", "CNN"],
             ["ARV", "WARV", "CNN"]
             ]

    dic = {}
    for c in pairs:
        results = []
        for i in range(30):
            valid, test = load(i, valid_label, c), load(i, test_label, c)
            l,a = deep(valid, test)
            results.append(tuple((l,a)))
        dic['-'.join(c)] = results

    for pair, results in dic.items():
        acc = []
        for r in results:
            acc.append(r[1])
            print(pair, "\t lost = {0} / acc = {1}".format(r[0], r[1]))

        np_acc = np.array(acc)
        print("Mean = " + str(np_acc.mean()))
        print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))
        print("\n")