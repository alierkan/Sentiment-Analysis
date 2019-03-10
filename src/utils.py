from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from nltk.tokenize import RegexpTokenizer
import csv
import os
import pandas as pd
import random
import math

tokenizer = RegexpTokenizer(r'\w+')
stopwords = []

def get_dataframe(ncols, nwin, file, start):
    if nwin is None:
        df = pd.read_csv(file, header=None, delimiter=',')
    else:
        df = pd.read_csv(file + str(ncols) + '_' + str(nwin) + '.txt', header=None, delimiter=',')
    labels = pd.DataFrame.as_matrix(df[start])
    data = pd.DataFrame.as_matrix(df.iloc[:, start+1:ncols+start+1])
    return labels, data


def normalize(data):
    for i,d in enumerate(data):
        data[i] = (d - d.min())/(d.max()-d.min())
    return data


def normalize2(data):
    for i, d in enumerate(data):
        norm = np.sqrt((d ** 2).sum())
        data[i] = d /norm
    return data


def get_word2vec_model(file, ncols, win):
    return KeyedVectors.load_word2vec_format(file + '_' + str(ncols) + '_' + str(win) + '.txt', binary=False)


def get_wordvector(model, token):
    try:
        return list(model.wv[token])
    except:
        return [0] * model.vector_size


def get_wordvectorNone(model, token):
    try:
        return list(model.wv[token])
    except:
        return None


def get_review_vector(model, tokens, ncols):
    review_vector = np.zeros(ncols)
    for token in tokens:
        review_vector = review_vector + np.array(get_wordvector(model, token))
    return review_vector


def get_review_vector_cross(model, tokens, ncols):
    review_vector = np.ones(ncols)
    for token in tokens:
        vector = get_wordvector(model, token)
        review_vector = np.cross(review_vector, vector)
    return review_vector


def get_glove_data(glove_dir, file):
    glove_dict = {}
    #glove_dir = '/home/alierkan/GloVe-1.2/'
    f = open(os.path.join(glove_dir, file))
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print(values[1:])
        norm = np.sqrt((coefs ** 2).sum())
        glove_dict[word] = coefs /norm
    f.close()
    print('Found %s word vectors.' % len(glove_dict))
    return glove_dict


def get_token_matrix(model, review, max_rlen, ncols, glove_dict, gm):
    token_list = np.zeros(shape=(max_rlen, gm*ncols))
    for t, token in enumerate(review):
        wv = np.asarray(get_wordvector(model, token))
        if gm > 1:
            glove = np.asarray(glove_dict.get(token, np.zeros(ncols)))
            wv = np.append(wv, glove)
        token_list[t] = wv
    return token_list


def get_token_matrix_weight(model, review, max_rlen, ncols, glove_dict, gm, word_dict):
    token_list = np.zeros(shape=(max_rlen, gm*ncols))
    for t, token in enumerate(review):
        mul = word_dict.get(token, 1)
        wv = np.asarray(get_wordvector(model, token))
        if gm > 1:
            glove = np.asarray(glove_dict.get(token, np.zeros(ncols)))
            wv = np.append(mul*wv, mul*glove)
        token_list[t] = wv
    return token_list


def get_stopwords():
    wl = []
    with open('../../resources/stopwords.csv') as csvfile:
        stopwords = csv.reader(csvfile, delimiter=',')
        for words in stopwords:
            for w in words:
                wl.append(str.strip(w))
    return wl


def get_word_counts(file, *args):
    word_dict = {}
    #if args:
    with open(file) as csvfile:
        words = csv.reader(csvfile, delimiter=',')
        if len(args) == 0:
            p, ne, nu = [0,0,0]
        else:
            p, ne, nu = args
        for i, word in enumerate(words):
            if i == 0 and len(args) == 0:
                word = list(map(float, word))
                p, ne, nu = word
                m = min(p, ne ,nu)
                p = p / m
                ne = ne / m
                nu = nu / m
            else:
                wordn = list(map(float, word[1:]))
                pos, neg, neu = wordn
                # pos += 1
                # neg += 1
                # neu += 1
                # all = pos + neg + neu
                # pos = all/pos
                # neg = all/neg
                # neu = all/neu
                #average = (pos + neg + neu) / 3 + 1
                #word_dict[word[0]] = 1 # No Weight
                word_dict[word[0]] = math.log2((pos/p + 1) / (neg/ne + 1)) #imdb
                #word_dict[word[0]] = math.log2(max((pos+1) / (neg+1), (neg+1) / (pos+1)))
                #word_dict[word[0]] = math.log2((pos+1) / (neg+1))
                #word_dict[word[0]] = math.log2(max((pos / p + 1) / (neg / ne + 1), (neg / ne + 1) / (pos / p + 1))) #semeval
                #print(str(pos) + ' - ' + str(neg) + ' - ' + str(neu) + ' - ' + str(average))
                #word_dict[word[0]] = math.log2(math.sqrt((pos - average)**2 + (neg - average)**2 + (neu - average)**2))
                #word_dict[word[0]] = math.log2((pos / p - average) ** 2 + (neg / ne - average) ** 2 + (neu / nu - average) ** 2 + 1)
    # else:
    #     with open(file) as csvfile:
    #         words = csv.reader(csvfile, delimiter=',')
    #         for word in words:
    #             pos = int(word[1])
    #             neg = int(word[2])
    #             word_dict[word[0]] = math.log2(max((pos+1)/(neg+1),(pos+1)/(neg+1)))
    return word_dict


class MyReviews(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield tokenizer.tokenize(line.lower())


def remove_stopwords(words, stopwords):
    for word in words:
        if word in stopwords:
            words.remove(word)
    return words


def get_list(input_file):
    stopwords = get_stopwords()
    clist = []

    rew_list = MyReviews(input_file)
    for words in rew_list:
        words = remove_stopwords(words, stopwords)
        label = words[-1]
        del words[-1]
        clist.append((words, label))
    return clist


def get_shuffle_list(file_pos=None, file_neg=None, shuffle=True):
    stopwords = get_stopwords()
    clist = []

    if file_pos:
        list_pos = MyReviews(file_pos)
        for words in list_pos:
            words = remove_stopwords(words, stopwords)
            clist.append((words, 1))

    if file_neg:
        list_neg = MyReviews(file_neg)
        for words in list_neg:
            words = remove_stopwords(words, stopwords)
            clist.append((words, 0))

    if shuffle:
        random.shuffle(clist)
    return clist


def get_shuffle_list_neutral(file_pos, file_neg, file_neu, shuffle):
    clist = []
    list_pos = MyReviews(file_pos)
    list_neg = MyReviews(file_neg)
    list_neu = MyReviews(file_neu)
    for t in list_pos:
        if t[0] not in stopwords:
            clist.append((t, [1,0,0]))
    for t in list_neg:
        if t[0] not in stopwords:
            clist.append((t, [0,0,1]))
    for t in list_neu:
        if t[0] not in stopwords:
            clist.append((t, [0,1,0]))
    if shuffle:
        random.shuffle(clist)
    return clist


def read_feature_matrix(dir, filename):
    x = np.genfromtxt(dir + "x_" + filename, delimiter=",")
    y = np.genfromtxt(dir + "y_" + filename, delimiter=",")
    return y,x


def write_score2(pred, classes, label, path = "/home/alierkan/phd/iclr15_IMDB/scores/"):
    with open(path+label, 'w') as out:
        for p, c in zip(pred, classes):
            out.write(str(c[0]) + " " + str(p[0]) + " " + str(1-p[0]) + "\n")


def write_score3(pred, classes, label, path = "/home/alierkan/phd/iclr15/scores/"):
    with open(path + label, 'w') as out:
        for p, c in zip(pred, classes):
            out.write(str(c) + " " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")


def write_score4(pred, label):
    with open("/home/alierkan/phd/iclr15/scores/"+label, 'w') as out:
        pred = np.ndarray.tolist(pred)
        for p in pred:
            out.write(str(p) + "\n")


def create_feature_matrix(wv_model, glove_dict, review_list, dir, filename:str, stop_words, word_dict, dim):
    y = np.zeros(shape=(len(review_list),dim))
    x = np.zeros(shape=(len(review_list),wv_model.syn0.shape[1] + glove_dict.get("a").shape[0]))
    for i,r in enumerate(review_list):
        rv = np.zeros(shape=(wv_model.syn0.shape[1]))
        gv = np.zeros(shape=(glove_dict.get("a").shape[0]))
        cntw = 0
        cntg = 0
        y[i] = r[1]
        for token in r[0]:
            if token not in stop_words:
                mul = word_dict.get(token,1)
                vw = get_wordvectorNone(wv_model, token)
                if vw:
                    rv = rv + mul*np.asarray(vw)
                    cntw += 1

                if token in glove_dict:
                    d = glove_dict[token]
                    norm = np.sqrt((d ** 2).sum())
                    gv = gv + mul*d/norm
                    cntg += 1
        if cntg == 0:
            cntg = 1
        if cntw == 0:
            cntw = 1
        x[i] = np.append(rv / cntw, gv /cntg)
        # x[i] = rv / cntw
        # x[i] = gv / cntg
    np.savetxt(dir + "x_" + filename, x, delimiter=",")
    np.savetxt(dir + "y_" + filename, y, delimiter=",")
    return y, x


if __name__ == "__main__":
    print(get_stopwords())