
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer.feature_transformer import FeatureTransformer
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer(stop_words= frozenset(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").read().splitlines()))),
            ("tfidf", TfidfTransformer()),
            ("clf", KNeighborsClassifier(n_neighbors=4))
        ])
        return pipe_line