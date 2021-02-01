from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer
from sklearn.linear_model import SGDClassifier


class SVMModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():



        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer(stop_words= frozenset(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines()))),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=None))
        ])

        return pipe_line