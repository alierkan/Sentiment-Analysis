import utils
import numpy as np
import keras
from keras.models import Sequential
from keras import layers, activations

num_classes = 3
gm = 2
ncols = 300
nwin = 5


def get_max_number_of_token():
    reviews = utils.MyReviews('../resources/semeval/data/full-train-all.txt')
    max_token = 0
    for review in reviews:
        length = len(review)
        if max_token < length:
            max_token = length
    return max_token


def get_review_windows(model, reviews, max_rlen, ncols, nsen, glove_dict):
    word_dict = utils.get_word_counts("../resources/semeval/data/word_counts.txt",6,2,1)
    x = np.zeros(shape=(nsen, max_rlen, gm*ncols))
    y = np.zeros(shape=(nsen,num_classes))

    for i,review in enumerate(reviews):
        try:
            #x[i] = utils.get_token_matrix(model, review[0], max_rlen, ncols, glove_dict, gm)
            x[i] = utils.get_token_matrix_weight(model, review[0], max_rlen, ncols, glove_dict, gm, word_dict)
        except IndexError as e:
            print(e)
        y[i] = review[1]

    x = x.reshape(x.shape[0], max_rlen, gm*ncols, 1)
    x = x.astype('float32')

    #y = keras_test.utils.to_categorical(y, num_classes)
    return x,y


def my_model(max_rlen):
    print("Longest sentence = " + str(max_rlen))
    print("Number of Word Vectors = " + str(ncols))
    input_shape = (max_rlen, gm*ncols, 1)
    n = 4

    model = Sequential()
    model.add(layers.Conv2D(filters=gm*ncols, kernel_size=(n, gm*ncols), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(max_rlen-n+1, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(max_rlen, activation=activations.relu))
    model.add(layers.Dense(num_classes, activation=activations.softmax))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
    return model


def main():
    vw_model = utils.get_word2vec_model('../resources/yelp/word2vec/yelp_restaurants_word2vector', ncols, nwin)
    vw_model.syn0 = utils.normalize2(vw_model.syn0)
    glove_dict = utils.get_glove_data('../resources/yelp/glove/','vectors_' + str(ncols) + '.txt')

    train_list = utils.get_shuffle_list_neutral('../resources/semeval/data/full-train-pos.txt',
                                                '../resources/semeval/data/full-train-neg.txt',
                                                '../resources/semeval/data/full-train-neu.txt', True)

    test_list = utils.get_shuffle_list_neutral('../resources/semeval/data/full-train-pos.txt',
                                                '../resources/semeval/data/full-train-neg.txt',
                                                '../resources/semeval/data/full-train-neu.txt', False)

    # test_list = utils.get_shuffle_list_neutral('../resources/semeval/data/test-pos.txt',
    #                                            '../resources/semeval/data/test-neg.txt',
    #                                            '../resources/semeval/data/test-neu.txt', False)

    max_rlen = get_max_number_of_token()

    x_train, y_train = get_review_windows(vw_model, train_list, max_rlen, ncols, len(train_list), glove_dict)
    x_test, y_test = get_review_windows(vw_model, test_list, max_rlen, ncols, len(test_list), glove_dict)

    results = []
    for i in range(30):
        model = my_model(max_rlen)
        history = model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1)
        score = model.evaluate(x_test, y_test, verbose=1)
        pred = model.predict(x_test, verbose=1)
        classes = model.predict_classes(x_test, verbose=1)
        utils.write_score3(pred, classes, "CNN-VALID_"+str(i))
        model = None
        results.append(score)

    acc = []
    for result in results:
        acc.append(result[1])
        print(result)

    np_acc = np.array(acc)
    print("Mean = " + str(np_acc.mean()))
    print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))


if __name__ == "__main__":
    main()