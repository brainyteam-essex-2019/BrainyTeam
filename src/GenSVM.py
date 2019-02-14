import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.svm import SVC

TEST_RATIO = 0.2


class SVMModel:

    def __init__(self):
        self.data = pd.DataFrame()
        self.target = pd.DataFrame()
        self.classifier = SVC()
        self.mae = 0
        self.mse = 0
        self.matrix = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data.head()
        return self.data

    def load_target(self, filepath):
        self.target = pd.read_csv(filepath)
        self.target.head()
        return self.target

    def get_sets(self):
        # TODO
        return

    def train(self, data=None, target=None):
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        train_data, train_target, test_data, test_target = train_test_split(data, target, test_size=TEST_RATIO)

        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        kernels = ["linear", "rbf"]

        parameters = [{"C": Cs, "gamma": gammas, "kernel": kernels}]

        clf = GridSearchCV(SVC, parameters, cv=5, n_jobs=-1)
        clf.fit(train_data, train_target)

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)

        # TODO Graphing results during training

        test_true, test_pred = test_target, clf.predict(test_data)

        mse = mean_squared_error(test_true, test_pred)
        mae = mean_absolute_error(test_true, test_pred)
        matrix = confusion_matrix(test_true, test_pred)

        print("\nMean Square Error:", mse)
        print("Mean Absolute Error:", mae)
        print("Confusion Matrix:\n", matrix)

        self.classifier = clf
        return clf

    def get_metrics(self):
        # TODO
        return

    def disp_confusion_matrix(self):
        print(self.matrix)
