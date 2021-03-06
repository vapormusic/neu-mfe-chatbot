import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from models.naive_bayes_model import NaiveBayesModel
from models.svm_model import SVMModel
from filtration.InputCleanup import InputCleanup
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import SGDClassifier


class TextClassification(object):
    def __init__(self):
        self.test = None
        self.gs_clf = None
        self.gs_clf_svm = None
        self.clf_nb = None
        self.clf_svm = None


    def get_train_data(self):
        print("Initializating...")
        # Tạo train data
        #url = 'https://raw.githubusercontent.com/vapormusic/neu-mfe-chatbot/main/question-intent.csv'
        df1 = pd.read_csv('../data/question-intent.csv')
        pd.DataFrame(df1)
        df_train = pd.DataFrame(df1)

        # Tokenizing
        i = 0
        for questions in df_train["feature"]:
            tokenized_questions = ViTokenizer.ViTokenizer.tokenize(questions)
            df_train["feature"][i] = tokenized_questions

        count_vect = CountVectorizer(open('../filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines())
        X_train_counts = count_vect.fit_transform(df_train["feature"])

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


        # NB models

        self.clf_nb = MultinomialNB()
        self.clf_nb.fit(X_train_tfidf, df_train.target)


        # SVM models

        self.clf_svm = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=None)
        self.clf_svm.fit(X_train_tfidf, df_train.target)

        # Grid search NB and SVM
          # Must use pipeline???
        model_nb = NaiveBayesModel()
        model_svm = SVMModel()


        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3),
                      }

        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                          'tfidf__use_idf': (True, False),
                          'clf-svm__alpha': (1e-2, 1e-3),
                          }

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        gs_clf = GridSearchCV(model_nb.clf, parameters, n_jobs=-1, cv=cv)
        self.gs_clf = gs_clf.fit(df_train["feature"], df_train.target)
        gs_clf_svm = GridSearchCV(model_svm.clf, parameters_svm, n_jobs=-1, cv=cv)
        self.gs_clf_svm = gs_clf_svm.fit(df_train["feature"], df_train.target)

        print("Initialization complete.")



    def test_data(self, input_string, expected_intent):
            test_data = []
            test_data.append({"feature": input_string, "target": expected_intent})
            df_test = pd.DataFrame(test_data)
            count_vect = CountVectorizer(open('../filtration/vietnamese-stopwords-dash.txt', encoding="utf8").readlines())
            X_train_counts = count_vect.fit_transform(df_test["feature"])
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
            predicted_nb = self.clf_nb.predict(X_train_tfidf)
            predicted_gs_svm = self.gs_clf_svm.predict(X_train_tfidf)
            predicted_gs_nb = self.gs_clf.predict(X_train_tfidf)
            predicted_svm = self.clf_svm.predict(X_train_tfidf)

            print("Naive Bayes Result: ")
            print(predicted_nb)
            print("SVM models result: ")
            print(predicted_svm)
            print("Grid Search Naive Bayes Result: ")
            print(predicted_gs_nb)
            print('Best Score: %s' % self.gs_clf.best_score_)
            print('Best Hyperparameters: %s' % self.gs_clf.best_params_)
            print("Grid Search SVM Result: ")
            print(predicted_gs_svm)
            print('Best Score: %s' % self.gs_clf_svm.best_score_)
            print('Best Hyperparameters: %s' % self.gs_clf_svm.best_params_)


if __name__ == '__main__':
    tcp = TextClassification()
    tcp.get_train_data()
    x = 1
    while x == 1:
        print("Câu hỏi của bạn : \n")
        string = str(input())
        #print("Expected intent : \n")
        #string2 = str(input())
        tokenized = ViTokenizer.ViTokenizer.tokenize(string)
        tcp.test_data(string, "")
