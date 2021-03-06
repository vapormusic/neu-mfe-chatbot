
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer
from sklearn.linear_model import LogisticRegression

class LogitModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer(stop_words= frozenset(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").read().splitlines()))),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000))
        ])
        return pipe_line