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
        output3 = self.stopwords_remove(output2)
        return output3

    def stopwords_remove(self,input_str):
        stopword_lists = open('filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines()
        for stopword in stopword_lists:
            input_str.replace(stopword, '')
        return input_str
