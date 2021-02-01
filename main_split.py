import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from model.naive_bayes_model import NaiveBayesModel
from model.svm_model import SVMModel
from model.logit_model import LogitModel
from filtration.InputCleanup import InputCleanup
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split


class TextClassification(object):
    def __init__(self):
        self.test = None
        self.gs_clf = None
        self.gs_clf_logit = None
        self.gs_clf_svm = None
        self.clf_nb = None
        self.clf_svm = None
        self.clf_logit = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def get_train_data(self):
        print("Initializating...")
        # Get data
        #url = 'https://raw.githubusercontent.com/vapormusic/neu-mfe-chatbot/main/question-intent.csv'
        df1 = pd.read_csv('question-intent.csv')
        pd.DataFrame(df1)
        df_orig = pd.DataFrame(df1)

        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_orig, df_orig.target, test_size=0.2)
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)

        # NB model
        model_nb = NaiveBayesModel()
        self.clf_nb = model_nb.clf.fit(self.X_train["feature"],self.y_train)
        self.clf_nb.score
        # SVM model

        model_svm = SVMModel()
        self.clf_svm = model_svm.clf.fit(self.X_train["feature"], self.y_train)

        # Logit model

        model_logit = LogitModel()
        self.clf_logit = model_logit.clf.fit(self.X_train["feature"], self.y_train)


        # Grid search NB and SVM

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3),
                      }

        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                          'tfidf__use_idf': (True, False),
                          'clf-svm__alpha': (1e-2, 1e-3),
                          }

        parameters_logit = {'vect__ngram_range': [(1, 1), (1, 2)],
                          'tfidf__use_idf': (True, False)
                          }

        cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=3, random_state=1)

        gs_clf = GridSearchCV(model_nb.clf, parameters, n_jobs=-1, cv=cv)
        self.gs_clf = gs_clf.fit(self.X_train["feature"], self.y_train)


        gs_clf_svm = GridSearchCV(model_svm.clf, parameters_svm, n_jobs=-1, cv=cv)
        self.gs_clf_svm = gs_clf_svm.fit(self.X_train["feature"], self.y_train)

        gs_clf_logit = GridSearchCV(model_logit.clf, parameters_logit, n_jobs=-1, cv=cv)
        self.gs_clf_logit = gs_clf_logit.fit(self.X_train["feature"], self.y_train)
        print("Initialization complete.")

        score_nb = self.clf_nb.score(self.X_test["feature"], self.y_test)
        score_svm = self.clf_svm.score(self.X_test["feature"], self.y_test)
        score_logit = self.clf_logit.score(self.X_test["feature"], self.y_test)
        score_gs = self.gs_clf.score(self.X_test["feature"], self.y_test)
        score_gs_svm = self.gs_clf_svm.score(self.X_test["feature"], self.y_test)
        score_gs_logit = self.gs_clf_logit.score(self.X_test["feature"], self.y_test)

        print("accuracy: ")
        print("NB: " + str(score_nb))
        print("SVM: " + str(score_svm))
        print("Logit: " + str(score_logit))
        print("GS_NB: " + str(score_gs))
        print("GS_SVM: " + str(score_gs_svm))
        print("GS_Logit: " + str(score_gs_logit))




    def test_data(self, input_string, expected_intent):
            test_data = []
            test_data.append({"feature": input_string, "target": expected_intent})
            df_test = pd.DataFrame(test_data)
            predicted_nb = self.clf_nb.predict(df_test["feature"])
            predicted_svm = self.clf_svm.predict(df_test["feature"])
            predicted_logit = self.clf_logit.predict(df_test["feature"])
            predicted_gs_svm = self.gs_clf_svm.predict(df_test["feature"])
            predicted_gs_nb = self.gs_clf.predict(df_test["feature"])
            predicted_gs_logit = self.gs_clf_logit.predict(df_test["feature"])
            # Tạo test data

            print("Naive Bayes Result: ")
            print(predicted_nb)
            print("SVM model result: ")
            print(predicted_svm)
            print("Logit model result: ")
            print(predicted_logit)
            print("Grid Search Naive Bayes Result: ")
            print(predicted_gs_nb)
            print('Best Score: %s' % self.gs_clf.best_score_)
            print('Best Hyperparameters: %s' % self.gs_clf.best_params_)
            print("Grid Search SVM Result: ")
            print(predicted_gs_svm)
            print('Best Score: %s' % self.gs_clf_svm.best_score_)
            print('Best Hyperparameters: %s' % self.gs_clf_svm.best_params_)
            print("Grid Search Logit Result: ")
            print(predicted_gs_logit)
            print('Best Score: %s' % self.gs_clf_logit.best_score_)
            print('Best Hyperparameters: %s' % self.gs_clf_logit.best_params_)



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
        tcp.test_data(InputCleanup().word_cleanup(string), "")
