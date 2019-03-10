import numpy as np
import os
import pandas as pd

n=30


def normalize(data):
    norm = np.sqrt((data ** 2).sum())
    return data / norm


def load(e, i, classifiers=["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN", "WARV"], path="../resources/imdb/scores"):
    """ returns an array containing the scores of the specified classifiers.
    Note that we center the scores to 0."""
    assert e in ["TEST", "VALID"]
    probas = []
    for c in classifiers:
        if c in ["RNNLM", "PARAGRAPH", "NBSVM"]:
            # data = np.loadtxt(os.path.join(path,"-".join([c, e])))
            data = pd.read_csv(os.path.join(path, "-".join([c, e])), delimiter=" ", header=None).values
        elif c in ["ARV", "WARV"]:
            # data = np.loadtxt(os.path.join(path, "_".join(["-".join([c, e]), str(i)])))
            data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i)])), delimiter=" ",
                               header=None).values
        else:
            # data = np.loadtxt(os.path.join(path, "_".join(["-".join([c, e]), str(i%5)])))
            data = pd.read_csv(os.path.join(path, "_".join(["-".join([c, e]), str(i % 5)])), delimiter=" ",
                               header=None).values
        if "RNNLM" in c:
            data = normalize(data[:,2] - 1)
        elif "PARAG" in c or "NBSVM" in c or "DL" in c or "CNN" in c or "RV" in c:
            data = normalize(data[:, 2] - 0.5)
        probas += [data]
    x = np.vstack(probas).T
    y = np.vstack([np.ones((x.shape[0]//2, 1)), np.zeros((x.shape[0]//2, 1))])
    return (x, y)


def accuracy(k, d):
    """ d is an array of shape nsamples, nclassifiers
    k is an array of size nclassifiers
    this function return the accuracy of the linear combination
    with k coefficients"""
    x, y = d
    output = [k[i] * x[:, i] for i in range(len(k))]
    pred = np.vstack(output).sum(0)
    cnt = ((pred < 0) == y.T).mean()
    return cnt * 100.


def ensemble(d, classifiers):
    """ computes the weigths of each ensemble
    according to the contribution of each model
    on the valid set """
    output = []
    for i, c in enumerate(classifiers):
        k = np.zeros(len(classifiers))
        k[i] = 1
        acc = accuracy(k, d)
        output += [acc]
    k = np.array(output)
    k /= k.sum()
    best = accuracy(k, d)
    return k, best


if __name__ == "__main__":
    dic = {}
    print("\n")
    # solo
    solos = [["RNNLM"], ["PARAGRAPH"], ["NBSVM"], ["ARV"], ["CNN"], ["WARV"]]
    for c in solos:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            vacc, tacc = accuracy([1], valid), accuracy([1], test)
            #print(c, "\t valid / test %.2f / %.2f" % (vacc, tacc))
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

    # duo
    duos = [["RNNLM", "PARAGRAPH"],
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
            ["WARV", "CNN"]
            ]
    for c in duos:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            k, vacc = ensemble(valid, c)
            tacc = accuracy(k, test)
            #print(c, "\t valid / test %.2f / %.2f - weigths =" % (vacc, tacc), k)
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

    # trios
    trios = [["RNNLM", "PARAGRAPH", "NBSVM"],
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
             ["RNNLM", "NBSVM", "WARV"]
             ]
    for c in trios:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            k, vacc = ensemble(valid, c)
            tacc = accuracy(k, test)
            #print(c, "\t valid / test %.2f / %.2f - weigths" % (vacc, tacc), k)
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

    fouros = [["RNNLM", "PARAGRAPH", "NBSVM", "ARV"],
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
              ["PARAGRAPH", "WARV", "ARV", "CNN"]
              ]
    for c in fouros:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            k, vacc = ensemble(valid, c)
            tacc = accuracy(k, test)
            #print(c, "\t valid / test %.2f / %.2f - weigths" % (vacc, tacc), k)
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

    fivos = [["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN"],
             ["WARV", "PARAGRAPH", "NBSVM", "ARV", "CNN"],
             ["RNNLM", "WARV", "NBSVM", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "WARV", "ARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "WARV", "CNN"],
             ["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "WARV"]
             ]
    for c in fivos:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            k, vacc = ensemble(valid, c)
            tacc = accuracy(k, test)
            #print(c, "\t valid / test %.2f / %.2f - weigths" % (vacc, tacc), k)
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

    sixos = [["RNNLM", "PARAGRAPH", "NBSVM", "ARV", "CNN", "WARV"]]
    for c in sixos:
        results = []
        for i in range(n):
            valid, test = load("VALID", i, c), load("TEST", i, c)
            k, vacc = ensemble(valid, c)
            tacc = accuracy(k, test)
            #print(c, "\t valid / test %.2f / %.2f - weigths" % (vacc, tacc), k)
            results.append(tuple((vacc, tacc)))
        #print("\n")
        dic['-'.join(c)] = results

for pair, results in dic.items():
    acc = []
    for r in results:
        acc.append(r[1])

    np_acc = np.array(acc)
    print(pair)
    print("Mean = " + str(np_acc.mean()))
    print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))
