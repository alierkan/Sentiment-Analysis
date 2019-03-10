# import modules & set up logging
import gensim
import logging
import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    reviews = utils.MyReviews('../resources/yelp/yelp_restaurants_review_sentences.txt')
    nsize = 50
    win = 5
    model = gensim.models.Word2Vec(reviews, size=nsize, window=win, min_count=5, workers=4, iter=25)
    model.save('../resources/yelp/word2vec/yelp_restaurants_word2vector' + '_' + str(nsize) + '_' + str(win) + '.bin')
    model.wv.save_word2vec_format('../resources/yelp/word2vec/yelp_restaurants_word2vector' + '_' + str(nsize) + '_' + str(win) + '.txt', binary=False)


if __name__ == "__main__":
    main()