import utils
from keras import models, layers, optimizers, losses, activations
import random
import numpy as np

nwin = 5
ncols = 200
ncols_mul = 2
total = 50000
div = 10
part = total//div


def get_wordvectors():
    print("wordvectors read ...")
    word_dict = utils.get_word_counts("/../resources/yelp/data/yelp_restaurant_word_counts.txt")
    stop_words = utils.get_stopwords()
    vw_model = utils.get_word2vec_model('../resources/yelp/word2vec/yelp_restaurants_word2vector', ncols, nwin)
    vw_model.syn0 = utils.normalize2(vw_model.syn0)
    glove_dict = utils.get_glove_data('../resources/yelp/glove/','vectors_'+str(ncols)+'.txt')
    return vw_model, word_dict, stop_words, glove_dict


def create_data(counter):
    print("create data ...= " + str(counter))
    list_pos = utils.get_shuffle_list('../resources/yelp/yelp_review_sentences_pos.txt', None, True)
    list_neg = utils.get_shuffle_list(None, '../resources/yelp/yelp_review_sentences_neg.txt', True)

    intervals = (counter*part, (counter+1)*part)

    train_list_pos = list_pos[0:intervals[0]] + list_pos[intervals[1]:total]
    train_list_neg = list_neg[0:intervals[0]] + list_neg[intervals[1]:total]
    test_list_pos = list_pos[intervals[0]:intervals[1]]
    test_list_neg = list_neg[intervals[0]:intervals[1]]

    random.shuffle(train_list_pos)
    random.shuffle(train_list_neg)
    train_list = train_list_pos + train_list_neg
    #random.shuffle(train_list)

    random.shuffle(test_list_pos)
    random.shuffle(test_list_neg)
    test_list = test_list_pos + test_list_neg
    #random.shuffle(test_list)

    with open("../resources/yelp/cv_train_data_" + str(counter) + ".txt", 'w') as out:
        for t in train_list:
            line = ""
            for l in t[0]:
                line = line + l + " "
            out.write(line + " " + str(t[1]) + "\n")

    with open("../resources/yelp/cv_test_data_" + str(counter) + ".txt", 'w') as out:
        for t in test_list:
            line = ""
            for l in t[0]:
                line = line + l + " "
            out.write(line + " " + str(t[1]) + "\n")

    print("files write ...")
    return train_list, test_list


def main():
    results = []
    vw_model, word_dict, stop_words, glove_dict = get_wordvectors()
    for i in range(div):
        #train_list, test_list = create_data(i)
        train_list = utils.get_list('../resources/yelp/cv_train_data_'+ str(i) +'.txt')
        test_list = utils.get_list('../resources/yelp/cv_test_data_'+ str(i) +'.txt')

        y_train, x_train = utils.create_feature_matrix(vw_model, glove_dict, train_list,
                                                       "../resources/yelp/data/",
                                                       "yelp_wvgl_train_cluster_normalize_"+ str(ncols) +"_"+str(nwin)+".txt", stop_words,
                                                       word_dict, 1)
        y_test, x_test = utils.create_feature_matrix(vw_model, glove_dict, test_list,
                                                     "../resources/yelp/data/",
                                                     "yelp_wvgl_test_cluster_normalize_" +str(ncols)+"_"+str(nwin)+"5.txt", stop_words,
                                                     word_dict, 1)

        model = models.Sequential()
        model.add(layers.Dense(ncols//2, activation=activations.relu, input_shape=(ncols * ncols_mul,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation=activations.sigmoid))
        model.compile(optimizer=optimizers.Adadelta(), loss=losses.binary_crossentropy, metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=10, batch_size=25)
        pred = model.predict_proba(x_test)
        classes = model.predict_classes(x_test)
        utils.write_score2(pred, classes, "WARV-TEST_"+str(i), "../resources/yelp/scores/")
        results.append(model.evaluate(x_test, y_test))

    acc = []
    for result in results:
        acc.append(result[1])
        print(result)

    np_acc = np.array(acc)
    print("Mean = " + str(np_acc.mean()))
    print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))


if __name__ == "__main__":
    main()