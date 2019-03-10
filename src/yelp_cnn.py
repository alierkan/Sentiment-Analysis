import utils
import numpy as np
import keras
from keras.models import Sequential
from keras import layers, activations
import random

num_classes = 2
gm = 2
nbatch = 1000
nepochs = 20
ncols = 200
nwin = 5

def get_max_number_of_token():
    reviews = utils.MyReviews('../resources/yelp/yelp_restaurants_review_sentences.txt')
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
        #x[i] = utils.get_token_matrix(model, review[0], max_rlen, ncols, glove_dict, gm)
        x[i] = utils.get_token_matrix_weight(model, review[0], max_rlen, ncols, glove_dict, gm, word_dict)
        y[i] = review[1]

    x = x.reshape(x.shape[0], max_rlen, gm*ncols, 1)
    x = x.astype('float32')

    y = keras.utils.to_categorical(y, num_classes)
    return x,y


def my_model(max_rlen):
    print("Longest sentence = " + str(max_rlen))
    print("Number of Word Vectors = " + str(ncols))
    input_shape = (max_rlen, gm*ncols, 1)
    n = max_rlen //2

    model = Sequential()
    model.add(layers.Conv2D(filters=10, kernel_size=(n, 10), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(max_rlen-n+1, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(max_rlen, activation=activations.relu))
    model.add(layers.Dense(num_classes, activation=activations.softmax))
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


def get_list():
    list_pos = utils.get_shuffle_list('../resources/yelp/yelp_review_sentences_pos.txt', None, True)
    list_neg = utils.get_shuffle_list(None, '../resources/yelp/yelp_review_sentences_neg.txt', True)

    part = 50000
    train_list_pos = list_pos[0:part]
    train_list_neg = list_neg[0:part]
    # test_list_pos = list_pos[0:part] #For Validation
    # test_list_neg = list_neg[0:part] #For Validation
    test_list_pos = list_pos[part:part*2]
    test_list_neg = list_neg[part:part*2]
    list_neg = []
    list_pos = []

    train_list = train_list_pos + train_list_neg
    #random.shuffle(train_list)

    test_list = test_list_pos + test_list_neg
    #random.shuffle(test_list)

    return train_list, test_list


def main():
    vw_model = utils.get_word2vec_model('../resources/yelp/word2vec/yelp_restaurants_word2vector', ncols, nwin)
    vw_model.vectors = utils.normalize2(vw_model.vectors)
    glove_dict = utils.get_glove_data('../resources/yelp/glove/', 'vectors_' + str(ncols) + '.txt')
    word_dict = utils.get_word_counts("../resources/yelp/data/yelp_restaurant_word_counts.txt")
    # train_list, test_list = get_list()
    # train_list = utils.get_list('../resources/yelp/train_data50000.txt')
    # test_list = utils.get_list('../resources/yelp/test_data50000.txt')

    max_rlen = get_max_number_of_token()
    model = my_model(max_rlen)

    print("#################### Iterations ################\n")
    results = []
    for j in range(1):
        acc_hist = {}
        train_list = utils.get_list('../resources/yelp/cv_train_data_'+str(j) +'.txt')
        test_list = utils.get_list('../resources/yelp/cv_train_data_'+str(j)+'.txt')
        half = len(test_list) // 2
        for e in range(nepochs):
            for i in range(int(len(train_list)/nbatch)):
                x_train, y_train = get_review_windows(vw_model, train_list[nbatch*i:nbatch*(i+1)], max_rlen, ncols, nbatch, glove_dict, word_dict)
                (loss,acc) = model.train_on_batch(x_train, y_train)

            print("Train: Epoch:" + str(e+1) + " Loss = " + str(loss) + " -- " + "Accuracy = " + str(acc))
            acc_hist[str(e+1)] = acc

        with open("../resources/yelp/scores/CNN-VALID_"+str(j), 'w') as out:
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
                    out.write(str(c) + " " + str(p[1]) + " " + str(p[0]) + "\n")

                print("Test: Iteration:" + str(i + 1) + " Loss = " + str(loss))
            results.append(score)
        print("######################## Trial = " + str(j))

    acc = []
    for result in results:
        acc.append(result)
        print(result)

    np_acc = np.array(acc)
    print("Mean = " + str(np_acc.mean()))
    print("Std.Dev = " + str(np.std(np_acc, dtype=np.float64)))


if __name__ == "__main__":
    main()