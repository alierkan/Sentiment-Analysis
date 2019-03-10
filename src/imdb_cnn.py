import utils
import numpy as np
import keras
from keras.models import Sequential
from keras import layers

num_classes = 2
gm = 1
nbatch = 100
nepochs = 20
ncols = 300
nwin = 5


def get_max_number_of_token():
    reviews = utils.MyReviews('../resources/imdb/data/alldata.txt')
    max_token = 0
    for review in reviews:
        length = len(review)
        if max_token < length:
            max_token = length
    return max_token


def get_review_windows(model, reviews, max_rlen, ncols, nsen, glove_dict, word_dict):
    x = np.zeros(shape=(nsen, max_rlen, gm*ncols))
    y = np.zeros(nsen)

    for i,review in enumerate(reviews):
        x[i] = utils.get_token_matrix_weight(model, review[0], max_rlen, ncols, glove_dict, gm, word_dict)
        y[i] = review[1]

    x = x.reshape(x.shape[0], max_rlen, gm*ncols, 1)
    x = x.astype('float32')

    y = keras.utils.to_categorical(y, num_classes)
    return x,y


def my_model(max_rlen):
    model = Sequential()
    print(max_rlen)
    print(ncols)
    input_shape = (max_rlen, gm*ncols, 1)
    n = 4

    model.add(layers.Conv2D(ncols, (n, ncols), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((max_rlen-n+1, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(max_rlen, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def main():

    vw_model = utils.get_word2vec_model('../resources/imdb/data/alldata_word2vector', ncols, nwin)
    vw_model.syn0 = utils.normalize2(vw_model.syn0)
    glove_dict = utils.get_glove_data('../resources/imdb/glove','vectors_300_5.txt')
    word_dict = utils.get_word_counts("../resources/imdb/data/word_counts.txt", 1, 1, 2)

    # train_list = utils.get_shuffle_list('../resources/imdb/data/full-train-pos.txt',
    #                                     '../resources/imdb/data/full-train-neg.txt', True)
    # test_list = utils.get_shuffle_list('../resources/imdb/data/test-pos.txt',
    #                                    '../resources/imdb/data/test-neg.txt', False)
    train_list = utils.get_shuffle_list('../resources/imdb/data/small-train-pos.txt','../resources/imdb/data/small-train-neg.txt', True)
    test_list = utils.get_shuffle_list('../resources/imdb/data/valid-pos.txt', '../resources/imdb/data/valid-neg.txt', False)
    max_rlen = get_max_number_of_token()
    model = my_model(max_rlen)

    half = len(test_list)//2
    results = []

    for j in range(30):
        acc_hist = {}
        for e in range(nepochs):
            for i in range(int(len(train_list)/nbatch)):
                x_train, y_train = get_review_windows(vw_model, train_list[nbatch*i:nbatch*(i+1)], max_rlen, ncols, nbatch, glove_dict, word_dict)
                (loss,acc) = model.train_on_batch(x_train, y_train)

            print("Train: Epoch:" + str(e+1) + " Loss = " + str(loss) + " -- " + "Accuracy = " + str(acc))
            acc_hist[str(e+1)] = acc

        with open("../resources/imdb/scores/CNN-VALID_"+str(j), 'w') as out:
            counter = 0
            score = 0
            for i in range(int(len(test_list)/nbatch)):
                x_test, y_test = get_review_windows(vw_model, test_list[nbatch*i:nbatch*(i+1)], max_rlen, ncols, nbatch, glove_dict, word_dict)
                loss = model.test_on_batch(x_test, y_test)
                pred = model.predict_proba(x_test)
                classes = model.predict_classes(x_test)
                counter += nbatch
                for p, c in zip(pred, classes):
                    # score:
                    if counter <= half and c == 1:
                        score += 1
                    elif counter > half and c == 0:
                        score += 1
                    out.write(str(c) + " " + str(p[1]) + " "+ str(p[0]) + "\n")

                print("Test: Iteration:" + str(i+1) + " Loss = " + str(loss))
            results.append(score)
        print("######################## Trial = "+ str(j))

    acc = []
    for result in results:
        acc.append(result)
        print(result)

    np_acc = np.array(acc)
    print("Mean = " + str(np_acc.mean()))
    print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))

if __name__ == "__main__":
    main()