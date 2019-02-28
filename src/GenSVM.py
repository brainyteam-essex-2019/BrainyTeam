import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, f1_score
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
        raise NotImplementedError

    def train(self, data=None, target=None):
        # uses own dataset and labels if none supplied
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        # splits data into training and testing datasets
        train_data, train_target, test_data, test_target = train_test_split(data, target, test_size=TEST_RATIO)

        # parameters used in Grid Search
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        kernels = ["linear", "rbf"]
        parameters = [{"C": Cs, "gamma": gammas, "kernel": kernels}]

        # trains SVM using Grid Search on training dataset
        clf = GridSearchCV(SVC, parameters, cv=5, n_jobs=-1)
        clf.fit(train_data, train_target)

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)

        # TODO Graphing results during training

        # predicts data on test dataset
        test_true, test_pred = test_target, clf.predict(test_data)

        # gets evaluation metrics from predictions
        mse = mean_squared_error(test_true, test_pred)
        mae = mean_absolute_error(test_true, test_pred)
        matrix = confusion_matrix(test_true, test_pred)
        f1 = f1_score(test_true, test_pred)

        # prints out metrics
        print("\nMean Square Error:", mse)
        print("Mean Absolute Error:", mae)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", matrix)

        # sets & returns trained SVM model
        self.classifier = clf
        return clf

    def get_metrics(self):
        # TODO
        NotImplementedError

    def disp_confusion_matrix(self):
        print(self.matrix)

    def get_results(self):
        return "MSE: {}, MAE: {}, F1 Score: {} Confussion Matrix: \n{}".format(self.mse, self.mae, self.f1, self.matrix)