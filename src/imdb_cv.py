import utils
from keras import models, layers, optimizers, losses, activations, metrics
import numpy as np


def main():
    ncols = 300
    ncols_mul = 2
    nwin = 5
    create = True
    word_dict = utils.get_word_counts("../resources/imdb/data/word_counts.txt", 1, 1, 2)
    if create:
        stop_words = utils.get_stopwords()
        vw_model = utils.get_word2vec_model('../resources/imdb/word2vec/alldata_word2vector', ncols, nwin)
        vw_model.syn0 = utils.normalize2(vw_model.syn0)
        glove_dict = utils.get_glove_data('../resources/imdb/glove','vectors_'+str(ncols)+'_'+str(nwin)+'.txt')

        # train_list = utils.get_shuffle_list('../resources/imdb/data/full-train-pos.txt',
        #                                     '../resources/imdb/data/full-train-neg.txt', True)
        # test_list = utils.get_shuffle_list('../resources/imdb/data/test-pos.txt',
        #                                    '../resources/imdb/data/test-neg.txt', False)
        train_list = utils.get_shuffle_list('../resources/imdb/data/small-train-pos.txt',
                                            '../resources/imdb/data/small-train-neg.txt', True)
        test_list = utils.get_shuffle_list('../resources/imdb/data/valid-pos.txt',
                                           '../resources/imdb/data/valid-neg.txt', False)
        print("files read ...")

        y_train, x_train = utils.create_feature_matrix(vw_model, glove_dict, train_list,
                                                 "../resources/imdb/data/",
                                                 "wvgl_train_cluster_normalize_200_"+str(ncols)+'_'+str(nwin)+'.txt', stop_words, word_dict, 1)
        y_test, x_test = utils.create_feature_matrix(vw_model, glove_dict, test_list,
                                               "../resources/imdb/data/",
                                               "wvgl_test_cluster_normalize_200_'+str(ncols)+'_'+str(nwin)+'.txt'", stop_words, word_dict, 1)
    else:
        y_train, x_train = utils.read_feature_matrix("../resources/imdb/data/","wvgl_train_cluster_normalize_"+str(ncols)+'_'+str(nwin)+'.txt')
        print("train list finished ...")
        y_test, x_test = utils.read_feature_matrix("../resources/imdb/data/","wvgl_test_cluster_normalize_"+str(ncols)+'_'+str(nwin)+'.txt')
        print("test list finished ...")


    results = []
    for i in range(30):
        model = models.Sequential()
        model.add(layers.Dense(ncols//4, activation=activations.relu, input_shape=(ncols * ncols_mul,)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1, activation=activations.sigmoid))
        model.compile(optimizer=optimizers.Adadelta(), loss=losses.binary_crossentropy, metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=100, batch_size=64)
        pred = model.predict_proba(x_test)
        classes = model.predict_classes(x_test)
        utils.write_score2(pred, classes, "WARV-VALID_"+str(i))
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