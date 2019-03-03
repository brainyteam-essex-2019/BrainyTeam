import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, f1_score, accuracy_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split

TEST_RATIO = 0.2


class LDAModel:
    def __init__(self):
        self.data = pd.DataFrame()
        self.target = pd.DataFrame()
        self.classifier = LDA()
        self.accuracy = 0
        self.balanced = 0
        self.mae = 0
        self.mse = 0
        self.f1 = 0
        self.matrix = None

    def load_data(self, data):
        self.data = data
        return self.data

    def load_target(self, target):
        self.target = target
        return self.target

    def get_metrics(self):
        print("Accuracy:", self.accuracy)
        print("Balanced accuracy:", self.balanced)
        print("Mean Square Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.matrix, "\n")

    def train(self, data=None, target=None):
        # uses own dataset and labels if none supplied
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        # splits data into training and testing datasets
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=TEST_RATIO)

        # train LDA model
        clf = LDA()
        clf.fit(train_data, train_target)

        # predicts data on test dataset
        test_true, test_pred = test_target, clf.predict(test_data)

        # gets evaluation metrics from predictions
        self.accuracy = accuracy_score(test_true, test_pred)
        self.balanced = balanced_accuracy_score(test_true, test_pred)
        self.mse = mean_squared_error(test_true, test_pred)
        self.mae = mean_absolute_error(test_true, test_pred)
        self.matrix = confusion_matrix(test_true, test_pred, labels=[1.0, 0.0])
        self.f1 = f1_score(test_true, test_pred)

        # prints out metrics
        self.get_metrics()

        # returns trained LDA model
        self.classifier = clf
        return clf

    def disp_confusion_matrix(self):
        print(self.matrix)
