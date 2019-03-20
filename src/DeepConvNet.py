import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, f1_score, accuracy_score, balanced_accuracy_score
from collections import Counter
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.utils import to_categorical


TEST_RATIO = 0.2


class CNNModel:

    def __init__(self):
        self.data = pd.DataFrame()
        self.target = pd.DataFrame()
        self.classifier = self.build_cnn()
        self.accuracy = 0
        self.balanced = 0
        self.mae = 0
        self.mse = 0
        self.f1 = 0
        self.matrix = None


    def build_cnn(self, ms=72, ch=64):

        model = Sequential()
        model.add(Reshape(target_shape=(1, ch, ms), input_shape=(ch, ms)))
        model.add(Conv2D(filters=25, kernel_size=(1, 5), activation='linear', padding='valid', data_format='channels_first'))
        model.add(Conv2D(filters=25, kernel_size=(ch, 1), activation='linear', padding='valid', data_format='channels_first'))
        model.add(BatchNormalization(epsilon=2.7**(-5), momentum=0.1))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=50, kernel_size=(1, 5), activation='linear', padding='valid', data_format='channels_first'))
        model.add(BatchNormalization(epsilon=2.7 ** (-5), momentum=0.1))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=100, kernel_size=(1, 5), activation='linear', padding='valid', data_format='channels_first'))
        model.add(BatchNormalization(epsilon=2.7 ** (-5), momentum=0.1))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=200, kernel_size=(1, 5), activation='linear', padding='valid', data_format='channels_first'))
        model.add(BatchNormalization(epsilon=2.7 ** (-5), momentum=0.1))
        model.add(Activation(activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(units=2, activation='softmax'))

        model.compile(optimizer='nadam', loss=['categorical_crossentropy'], metrics=['accuracy'])

        return model


    def load_data(self, data):
        self.data = data
        return self.data


    def load_target(self, target):
        self.target = target
        return self.target


    def get_sets(self):
        # TODO
        raise NotImplementedError


    def get_metrics(self):
        # TODO
        print("target {}".format(Counter(self.target)))
        print("Accuracy:", self.accuracy)
        print("Balanced accuracy:", self.balanced)
        print("Mean Square Error:", self.mse)
        print("Mean Absolute Error:", self.mae)
        print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.matrix,"\n")


    def train(self, data=None, target=None):
        # uses own dataset and labels if none supplied
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        # splits data into training and testing datasets
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=TEST_RATIO)

        #transform the targets in a one-hot array, this way problem is treated as classification and not regression
        train_target = to_categorical(train_target, num_classes=2)
        test_target = to_categorical(test_target, num_classes=2)

        print("train_data shape:", train_data.shape)
        print("train_target shape:", test_data.shape)
        print("test_data ahape:", train_target.shape)
        print("test_target shape:", test_target.shape)
        """

        # parameters used in Grid Search
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        kernels = ["linear", "rbf"]
        parameters = [{"C": Cs, "gamma": gammas, "kernel": kernels}]

        # trains SVM using Grid Search on training dataset
        clf = GridSearchCV(SVC(), parameters, cv=5, n_jobs=-1)
        """
        clf = self.classifier
        clf.fit(train_data, train_target, batch_size=1, validation_split=0.25, epochs=100)

        #print("Best parameters set found on development set:\n")
        #print(clf.best_params_)

        # TODO Graphing results during training

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

        # sets & returns trained SVM model
        self.classifier = clf
        return clf


    def disp_confusion_matrix(self):
        print(self.matrix)
