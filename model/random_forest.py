from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from transformer.feature_transformer import FeatureTransformer


class randomforest(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("vect", CountVectorizer(stop_words= frozenset(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines()))),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestRegressor(n_estimators = 100, random_state = 42))
        ])
        return pipe_line