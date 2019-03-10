import utils
from keras import models, layers, optimizers, losses, activations
import numpy as np

def main():
    ncols = 300
    ncols_mul = 2
    create = True
    word_dict = utils.get_word_counts("../resources/semeval/data/word_counts.txt", 6, 2, 1)
    if create:
        nwin = 5
        stop_words = utils.get_stopwords()
        vw_model = utils.get_word2vec_model('../resources/yelp/word2vec/yelp_restaurants_word2vector', ncols, nwin)
        vw_model.syn0 = utils.normalize2(vw_model.syn0)
        glove_dict = utils.get_glove_data('../resources/yelp/glove', 'vectors_' + str(ncols) + '.txt')

        train_list = utils.get_shuffle_list_neutral('../resources/semeval/data/full-train-pos.txt',
                                                    '../resources/semeval/data/full-train-neg.txt',
                                                    '../resources/semeval/data/full-train-neu.txt', True)
        test_list = utils.get_shuffle_list_neutral('../resources/semeval/data/full-train-pos.txt',
                                                    '../resources/semeval/data/full-train-neg.txt',
                                                    '../resources/semeval/data/full-train-neu.txt', False)
        # test_list = utils.get_shuffle_list_neutral('../resources/semeval/data/test-pos.txt',
        #                                         '../resources/semeval/data/test-neg.txt',
        #                                         '../resources/semeval/data/test-neu.txt', False)
        print("files read ...")

        y_train, x_train = utils.create_feature_matrix(vw_model, glove_dict, train_list,
                                                 "../resources/semeval/data/",
                                                 "wvgl_train_cluster_normalize_300_5_neu.txt", stop_words, word_dict, 3)
        y_test, x_test = utils.create_feature_matrix(vw_model, glove_dict, test_list,
                                               "../resources/semeval/data/",
                                               "wvgl_test_cluster_normalize_300_5_neu.txt", stop_words, word_dict, 3)
    else:
        y_train, x_train = utils.read_feature_matrix("../resources/semeval/data/","wvgl_train_cluster_normalize_300_5_neu.txt")
        print("train list finished ...")
        y_test, x_test = utils.read_feature_matrix("../resources/semeval/data/","wvgl_test_cluster_normalize_300_5_neu.txt")
        print("test list finished ...")

    results = []
    for i in range(30):
        model = models.Sequential()
        model.add(layers.Dense(ncols//2, activation=activations.relu, input_shape=(ncols * ncols_mul,)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(3, activation=activations.sigmoid))
        model.compile(optimizer=optimizers.Adadelta(), loss=losses.categorical_crossentropy, metrics=['categorical_accuracy'])
        history = model.fit(x_train, y_train, epochs=25, batch_size=25)
        pred = model.predict_proba(x_test)
        classes = model.predict_classes(x_test)
        utils.write_score3(pred, classes, "WARV-VALID_"+str(i))
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