import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from model.naive_bayes_model import NaiveBayesModel
from model.svm_model import SVMModel
from filtration.InputCleanup import InputCleanup
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


class TextClassification(object):
    def __init__(self):
        self.test = None

    def get_train_data(self, input_string, expected_intent):
        # Tạo train data
        #url = 'https://raw.githubusercontent.com/vapormusic/neu-mfe-chatbot/main/question-intent.csv'
        df1 = pd.read_csv('question-intent.csv')
        pd.DataFrame(df1)

        df_train = pd.DataFrame(df1)

        # Tạo test data
        test_data = []
        test_data.append({"feature": input_string, "target": expected_intent})
        df_test = pd.DataFrame(test_data)

        # NB model

        model_nb = NaiveBayesModel()

        clf_nb = model_nb.clf.fit(df_train["feature"], df_train.target)

        predicted_nb = clf_nb.predict(df_test["feature"])

        # SVM model

        model_svm = SVMModel()

        clf_svm = model_svm.clf.fit(df_train["feature"], df_train.target)

        predicted_svm = clf_svm.predict(df_test["feature"])

        # Grid search NB and SVM

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
        gs_clf = gs_clf.fit(df_train["feature"], df_train.target)
        predicted_gs_nb = gs_clf.predict(df_test["feature"])

        gs_clf_svm = GridSearchCV(model_svm.clf, parameters_svm, n_jobs=-1, cv=cv)
        gs_clf_svm = gs_clf_svm.fit(df_train["feature"], df_train.target)
        predicted_gs_svm = gs_clf_svm.predict(df_test["feature"])

        # Print predicted result
        print("Naive Bayes Result: ")
        print(predicted_nb)
        print("SVM model result: ")
        print(predicted_svm)
        print("Grid Search Naive Bayes Result: ")
        print(predicted_gs_nb)
        print('Best Score: %s' % gs_clf.best_score_)
        print('Best Hyperparameters: %s' % gs_clf.best_params_)
        print("Grid Search SVM Result: ")
        print(predicted_gs_svm)
        print('Best Score: %s' % gs_clf_svm.best_score_)
        print('Best Hyperparameters: %s' % gs_clf_svm.best_params_)
        # print(np.mean(gs_clf_svm.predict(df_test["feature"]) == df_test["target"]))


if __name__ == '__main__':
    x = 1
    while x == 1:
        print("Câu hỏi của bạn : \n")
        string = str(input())
        #print("Expected intent : \n")
        #string2 = str(input())
        tokenized = ViTokenizer.ViTokenizer.tokenize(string)
        tcp = TextClassification()
        tcp.get_train_data(InputCleanup().word_cleanup(string), "")
