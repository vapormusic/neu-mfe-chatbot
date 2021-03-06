import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from sklearn.ensemble import RandomForestRegressor

from models.decision_tree_model import decision_tree
from models.knn_model import KNNModel
from models.naive_bayes_model import NaiveBayesModel
from models.nbsvm_model import NBSVMModel
from models.random_forest import randomforest
from models.svm_model import SVMModel
from models.logit_model import LogitModel
from filtration.InputCleanup import InputCleanup
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import time


class TextClassification(object):
    def __init__(self):
        self.test = None
        self.gs_clf = None
        self.gs_clf_logit = None
        self.gs_clf_svm = None
        self.clf_knn = None
        self.clf_nbsvm = None
        self.clf_nb = None
        self.clf_svm = None
        self.clf_logit = None
        self.clf_decision_tree = None
        self.clf_random_forest = None
        self.clf_logit = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def get_train_data(self):
        print("Initializating...")
        # Get data
        #url = 'https://raw.githubusercontent.com/vapormusic/neu-mfe-chatbot/main/question-intent.csv'
        df1 = pd.read_csv('data/question-intent 06032021.csv')
        pd.DataFrame(df1)
        df_orig = pd.DataFrame(df1)

        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_orig, df_orig.target, test_size=0.3, random_state=1)
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)
        
        # NB model
        start = time.time()
        model_nb = NaiveBayesModel()
        self.clf_nb = model_nb.clf.fit(self.X_train["feature"], self.y_train)
        score_nb = self.clf_nb.score(self.X_test["feature"], self.y_test)
        elapsed_nb = (time.time() - start)

        # SVM model
        start = time.time()
        model_svm = SVMModel()
        self.clf_svm = model_svm.clf.fit(self.X_train["feature"], self.y_train)
        score_svm = self.clf_svm.score(self.X_test["feature"], self.y_test)
        elapsed_svm = (time.time() - start)

        # Logit model
        start = time.time()
        model_logit = LogitModel()
        self.clf_logit = model_logit.clf.fit(self.X_train["feature"], self.y_train)
        score_logit = self.clf_logit.score(self.X_test["feature"], self.y_test)
        elapsed_logit = (time.time() - start)


        #KNN
        start = time.time()
        knn = KNNModel()
        self.clf_knn = knn.clf.fit(self.X_train["feature"], self.y_train)
        score_knn = self.clf_knn.score(self.X_test["feature"], self.y_test)
        elapsed_knn = (time.time() - start)

        #NBSVM
        start = time.time()
        nbsvm = NBSVMModel()
        self.clf_nbsvm = nbsvm.clf.fit(self.X_train["feature"], self.y_train)
        score_nbsvm = self.clf_nbsvm.score(self.X_test["feature"], self.y_test)
        elapsed_nbsvm = (time.time() - start)

        #Decision Tree
        start = time.time()
        decisiontree = decision_tree()
        self.clf_decision_tree = decisiontree.clf.fit(self.X_train["feature"], self.y_train)
        score_decision_tree = self.clf_decision_tree.score(self.X_test["feature"], self.y_test)
        elapsed_decision_tree = (time.time() - start)

        #Random Forest
        start = time.time()
        random_forest = randomforest()
        le = preprocessing.LabelEncoder()
        Y_train_vec = le.fit_transform(self.y_train)
        Y_test_vec = le.fit_transform(self.y_test)
        self.clf_random_forest = random_forest.clf.fit(self.X_train["feature"], Y_train_vec)
        score_random_forest = self.clf_random_forest.score(self.X_test["feature"], Y_test_vec)
        elapsed_random_forest = (time.time() - start)

        # Grid search NB and SVM

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3),
                      }

        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                          'tfidf__use_idf': (True, False),
                          'clf-svm__alpha': (1e-3, 1e-2, 1e-1),
                      }

        parameters_logit = {'vect__ngram_range': [(1, 1), (1, 2)],
                          'tfidf__use_idf': (True, False)
                      }

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

        start = time.time()
        gs_clf = GridSearchCV(model_nb.clf, parameters, n_jobs=-1, cv=cv)
        self.gs_clf = gs_clf.fit(self.X_train["feature"], self.y_train)
        score_gs = self.gs_clf.score(self.X_test["feature"], self.y_test)
        elapsed_gs_nb = (time.time() - start)

        start = time.time()
        gs_clf_svm = GridSearchCV(model_svm.clf, parameters_svm, n_jobs=-1, cv=cv)
        self.gs_clf_svm = gs_clf_svm.fit(self.X_train["feature"], self.y_train)
        score_gs_svm = self.gs_clf_svm.score(self.X_test["feature"], self.y_test)
        elapsed_gs_svm = (time.time() - start)

        start = time.time()
        gs_clf_logit = GridSearchCV(model_logit.clf, parameters_logit, n_jobs=-1, cv=cv)
        self.gs_clf_logit = gs_clf_logit.fit(self.X_train["feature"], self.y_train)
        score_gs_logit = self.gs_clf_logit.score(self.X_test["feature"], self.y_test)
        elapsed_gs_logit = (time.time() - start)
        print("Initialization complete.")


        print("finish time:")
        print("NB: " + str(round(elapsed_nb, 4))+"s")
        print("SVM: " + str(round(elapsed_svm, 4))+"s")
        print("NBSVM: " + str(round(elapsed_nbsvm, 4)) + "s")
        print("Logit: " + str(round(elapsed_logit, 4))+"s")
        print("KNN: " + str(round(elapsed_knn, 4)) + "s")
        print("Random Forest: " + str(round(elapsed_random_forest, 4)) + "s")
        print("Decision Tree: " + str(round(elapsed_decision_tree, 4)) + "s")
        print("GS_NB: " + str(round(elapsed_gs_nb, 4))+"s")
        print("GS_SVM: " + str(round(elapsed_gs_svm, 4))+"s")
        print("GS_Logit: " + str(round(elapsed_gs_logit, 4))+"s")

        print("accuracy: ")
        print("NB: " + str(score_nb))
        print("SVM: " + str(score_svm))
        print("NBSVM: " + str(score_nbsvm))
        print("Logit: " + str(score_logit))
        print("KNN: " + str(score_knn))
        print("Random Forest: " + str(score_random_forest))
        print("Decision Tree: " + str(score_decision_tree))
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
            predicted_knn = self.clf_knn.predict(df_test["feature"])
            predicted_nbsvm = self.clf_nbsvm.predict(df_test["feature"])
            predicted_decision_tree = self.clf_decision_tree.predict(df_test["feature"])
            predicted_random_forest = self.clf_random_forest.predict(df_test["feature"])
            predicted_gs_svm = self.gs_clf_svm.predict(df_test["feature"])
            predicted_gs_nb = self.gs_clf.predict(df_test["feature"])
            predicted_gs_logit = self.gs_clf_logit.predict(df_test["feature"])
            # Tạo test data

            print("Naive Bayes Result: ")
            print(predicted_nb)
            print("SVM model result: ")
            print(predicted_svm)
            print("NBSVM model result: ")
            print(predicted_nbsvm)
            print("Logit model result: ")
            print(predicted_logit)
            print("KNN model result: ")
            print(predicted_knn)
            print("Decision Tree Result: ")
            print(predicted_decision_tree)
            print("Random Forest Result: ")
            print(predicted_random_forest)
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
