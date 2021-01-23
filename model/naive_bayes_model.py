from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer


class NaiveBayesModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines())),#bag-of-words
            ("tfidf", TfidfTransformer()),#tf-idf
            ("clf", MultinomialNB())#model naive bayes
        ])

        return pipe_line