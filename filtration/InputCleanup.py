from pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator
from pyvi import ViUtils
from underthesea import sent_tokenize



class InputCleanup:
    def __init__(self):
        print("")

    def word_cleanup(self, input_str):
        output = input_str.lower()
        output2 = ViUtils.add_accents(output)
        return output2


