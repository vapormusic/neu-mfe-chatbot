import pandas as pd

from model.naive_bayes_model import NaiveBayesModel
from model.svm_model  import SVMModel

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self, inputstring):
        # Tạo train data
        url = 'https://raw.githubusercontent.com/vapormusic/neu-mfe-chatbot/main/question-intent.csv'
        df1 = pd.read_csv(url)
        pd.DataFrame(df1)

        df_train = pd.DataFrame(df1)

        # Tạo test data
        test_data = []
        test_data.append({"feature": inputstring, "target": "Olympia"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel()

        clf = model.clf.fit(df_train["feature"], df_train.target)

        predicted = clf.predict(df_test["feature"])

        # Print predicted result
        print(predicted)
        print(clf.predict_proba(df_test["feature"]))


if __name__ == '__main__':
    x = 1
    while x == 1:
        print("Input please: \n")
        string = str(input())
        tcp = TextClassificationPredict()
        tcp.get_train_data(string)