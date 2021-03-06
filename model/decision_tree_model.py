from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from transformer.feature_transformer import FeatureTransformer
from sklearn import tree

class decision_tree(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer(stop_words= frozenset(open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines()))),
            ("tfidf", TfidfTransformer()),
            ("clf", tree.DecisionTreeClassifier())
        ])
        return pipe_line