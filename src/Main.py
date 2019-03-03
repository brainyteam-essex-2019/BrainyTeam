import time

import ModelSaver
import featureExtraction
# import PreProcessing
import numpy as np
from GenSVM import SVMModel
from LDAClassifier import LDAModel
from UTIL import ReadData

# import CommonSpatialPattern

"""
Inputs needed
1: Retrain models or not 
2: use PCA or ICA
3: What Window size to use? 
"""

LDA_PICKLE_FILE = "models/LDA_MODEL"
SVM_PICKLE_FILE = "models/SVM_MODEL"

SVM_NONE_FILE = SVM_PICKLE_FILE + "_NONE.pkl"
SVM_ICA_FILE = SVM_PICKLE_FILE + "_ICA.pkl"
SVM_PCA_FILE = SVM_PICKLE_FILE + "_PCA.pkl"
SVM_CSP_FILE = SVM_PICKLE_FILE + "_CSP.pkl"
SVM_ICA_PCA_FILE = SVM_PICKLE_FILE + "_ICA_PCA.pkl"
SVM_ICA_CSP_FILE = SVM_PICKLE_FILE + "_ICA_CSP.pkl"
SVM_PCA_CSP_FILE = SVM_PICKLE_FILE + "_PCA_CSP.pkl"
SVM_ICA_PCA_CSP_FILE = SVM_PICKLE_FILE + "_ICA_PCA_CSP.pkl"

LDA_NONE_FILE = LDA_PICKLE_FILE + "_NONE.pkl"
LDA_ICA_FILE = LDA_PICKLE_FILE + "_ICA.pkl"
LDA_PCA_FILE = LDA_PICKLE_FILE + "_PCA.pkl"
LDA_CSP_FILE = LDA_PICKLE_FILE + "_CSP.pkl"
LDA_ICA_PCA_FILE = LDA_PICKLE_FILE + "_ICA_PCA.pkl"
LDA_ICA_CSP_FILE = LDA_PICKLE_FILE + "_ICA_CSP.pkl"
LDA_PCA_CSP_FILE = LDA_PICKLE_FILE + "_PCA_CSP.pkl"
LDA_ICA_PCA_CSP_FILE = LDA_PICKLE_FILE + "_ICA_PCA_CSP.pkl"

# variables set by user
# pre_processing = True
retrain_models = False
window_size = 300
ica_components = 30
pca_components = 10

# models
svm = SVMModel()
lda = LDAModel()


def flatten_data(matrix_3d):
    new_matirx = np.empty((len(matrix_3d), len(matrix_3d[0]) * len(matrix_3d[0][0])))
    for i in range(len(matrix_3d)):
        matrix_2d = matrix_3d[i].flatten()
        new_matirx[i] = matrix_2d
    return new_matirx


def train_models():
    data, targets = ReadData()
    data_pca, targets_pca = featureExtraction.applyPCA(pca_components, data, targets)
    data_ica, targets_ica = featureExtraction.applyICA(ica_components, data, targets)
    data_features_csp, targets_csp = featureExtraction.analyzeCSP(data, targets)
    print("ICA Components ", ica_components)
    print("PCA Components ", pca_components)


    print("Features shape:", data.shape)
    print("Features targets:", data.shape)
    data_features_none = flatten_data(data)
    svm_none = train_model(svm, data_features_none, targets, SVM_NONE_FILE)

    print("Features shape:", data_ica.shape)
    print("Features targets:", targets.shape)
    data_features_ica = flatten_data(data_ica)
    svm_ica = train_model(svm, data_features_ica, targets, SVM_ICA_FILE)

    print("Features shape:", data_pca.shape)
    print("Features targets:", targets.shape)
    data_features_pca = flatten_data(data_pca)
    svm_pca = train_model(svm, data_features_pca, targets, SVM_PCA_FILE)

    print("Features shape:", data_features_csp.shape)
    print("Features targets:", targets.shape)
    svm_csp = train_model(svm, data_features_csp, targets, SVM_CSP_FILE)

    """
    print("Features shape:", data_ica_pca.shape)
    print("Features targets:", targets.shape)
    data_features_ica_pca = flatten_data(data_ica_pca)
    svm_ica_pca = train_model(svm, data_features_ica_pca, targets, SVM_ICA_PCA_FILE)
    """

    print("Features shape:", data_features_none.shape)
    print("Features targets:", targets.shape)
    lda_none = train_model(lda, data_features_none, targets, LDA_NONE_FILE)

    print("Features shape:", data_features_ica.shape)
    print("Features targets:", targets.shape)
    lda_ica = train_model(lda, data_features_ica, targets, LDA_ICA_FILE)

    print("Features shape:", data_features_pca.shape)
    print("Features targets:", targets.shape)
    lda_pca = train_model(lda, data_features_pca, targets, LDA_PCA_FILE)

    print("Features shape:", data_features_csp.shape)
    print("Features targets:", targets.shape)
    lda_csp = train_model(lda, data_features_csp, targets, LDA_CSP_FILE)

    """
    print("Features shape:", data_features_ica_pca.shape)
    print("Features targets:", targets.shape)
    lda_ica_pca = train_model(lda, data_features_ica_pca, targets, LDA_ICA_PCA_FILE)
    """


def train_model(model, features, targets, filepath):
    print("Training", filepath)
    t1 = time.time()
    model.load_data(features)
    model.load_target(targets)
    model.train()
    ModelSaver.save_model(model, filepath)
    t2 = time.time()
    print("Time to train was", t2 - t1)
    return model


def read_models():
    svm_none = ModelSaver.read_model(SVM_NONE_FILE)
    print("SVM None")
    print(svm_none.get_metrics())

    svm_ica = ModelSaver.read_model(SVM_ICA_FILE)
    print("SVM ICA")
    print(svm_ica.get_metrics())

    svm_pca = ModelSaver.read_model(SVM_PCA_FILE)
    print("SVM PCA")
    print(svm_pca.get_metrics())

    svm_csp = ModelSaver.read_model(SVM_CSP_FILE)
    print("SVM CSP")
    print(svm_csp.get_metrics())

    """
    svm_ica_pca = ModelSaver.read_model(SVM_ICA_PCA_FILE)
    print("SVM ICA PCA")
    print(svm_ica_pca.get_metrics())
    """

    lda_none = ModelSaver.read_model(LDA_NONE_FILE)
    print("LDA None")
    print(lda_none.get_metrics())

    lda_ica = ModelSaver.read_model(LDA_ICA_FILE)
    print("LDA ICA ")
    print(lda_ica.get_metrics())

    lda_pca = ModelSaver.read_model(LDA_PCA_FILE)
    print("LDA PCA")
    print(lda_pca.get_metrics())

    lda_csp = ModelSaver.read_model(LDA_CSP_FILE)
    print("LDA CSP")
    print(lda_csp.get_metrics())

    """
    lda_ica_pca = ModelSaver.read_model(LDA_ICA_PCA_FILE)
    print("LDA ICA PCA")
    print(lda_ica_pca.get_metrics())
    """


if retrain_models:
    train_models()
else:
    read_models()
