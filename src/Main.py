import time

import ModelSaver
import featureExtraction
# import PreProcessing
import numpy as np
from GenSVM import SVMModel
from LDAClassifier import LDAModel

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
retrain_models = True
use_ica = True
use_pca = False
use_csp = True
window_size = 300

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
    data_features_none, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    print("Features shape:", data_features_none.shape)
    print("Features targets:", data_targets.shape)
    data_features_none = flatten_data(data_features_none)
    svm_none = train_model(svm, data_features_none, data_targets, SVM_NONE_FILE)
    print(svm_none.get_results())

    data_features_ica, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=False)
    print("Features shape:", data_features_ica.shape)
    print("Features targets:", data_targets.shape)
    data_features_ica = flatten_data(data_features_ica)
    svm_ica = train_model(svm, data_features_ica, data_targets, SVM_ICA_FILE)
    print(svm_ica.get_results())

    data_features_pca, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=True)
    print("Features shape:", data_features_pca.shape)
    print("Features targets:", data_targets.shape)
    data_features_pca = flatten_data(data_features_pca)
    svm_pca = train_model(svm, data_features_pca, data_targets, SVM_PCA_FILE)
    print(svm_pca.get_results())

    data_features_ica_pca, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=True)
    print("Features shape:", data_features_ica_pca.shape)
    print("Features targets:", data_targets.shape)
    data_features_ica_pca = flatten_data(data_features_ica_pca)
    svm_ica_pca = train_model(svm, data_features_ica_pca, data_targets, SVM_ICA_PCA_FILE)
    print(svm_ica_pca.get_results())

    #data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    print("Features shape:", data_features_none.shape)
    print("Features targets:", data_targets.shape)
    lda_none = train_model(lda, data_features_none, data_targets, LDA_NONE_FILE)
    print(lda_none.get_results())

    #data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=False)
    print("Features shape:", data_features_ica.shape)
    print("Features targets:", data_targets.shape)
    lda_ica = train_model(lda, data_features_ica, data_targets, LDA_ICA_FILE)
    print(lda_ica.get_results())

    #data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=True)
    print("Features shape:", data_features_pca.shape)
    print("Features targets:", data_targets.shape)
    lda_pca = train_model(lda, data_features_pca, data_targets, LDA_PCA_FILE)
    print(lda_pca.get_results())

    #data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=True)
    print("Features shape:", data_features_ica_pca.shape)
    print("Features targets:", data_targets.shape)
    lda_ica_pca = train_model(lda, data_features_ica_pca, data_targets, LDA_ICA_PCA_FILE)
    print(lda_ica_pca.get_results())


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
    # svm_none = ModelSaver.read_model(SVM_NONE_FILE)
    # print(svm_none.get_results())

    svm_ica = ModelSaver.read_model(SVM_ICA_FILE)
    print(svm_ica.get_results())

    svm_pca = ModelSaver.read_model(SVM_PCA_FILE)
    print(svm_pca.get_results())

    svm_ica_pca = ModelSaver.read_model(SVM_CSP_FILE)
    print(svm_ica_pca.get_results())

    # lda_none = ModelSaver.read_model(LDA_NONE_FILE)
    # print(lda_none.get_results())

    lda_ica = ModelSaver.read_model(LDA_ICA_FILE)
    print(lda_ica.get_results())

    lda_pca = ModelSaver.read_model(LDA_PCA_FILE)
    print(lda_pca.get_results())

    lda_ica_pca = ModelSaver.read_model(LDA_ICA_PCA_FILE)
    print(lda_ica_pca.get_results())


if retrain_models:
    train_models()
else:
    read_models()
