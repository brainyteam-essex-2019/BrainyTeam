import featureExtraction
# import PreProcessing
import time
import ModelSaver
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


def train_models():
    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    svm_none = train_model(svm, data_features, data_targets, SVM_NONE_FILE)
    print(svm_none.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=False)
    svm_ica = train_model(svm, data_features, data_targets, SVM_ICA_FILE)
    print(svm_ica.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=True)
    svm_pca = train_model(svm, data_features, data_targets, SVM_PCA_FILE)
    print(svm_pca.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    svm_csp = train_model(svm, data_features, data_targets, SVM_CSP_FILE)
    print(svm_csp.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    lda_none = train_model(lda, data_features, data_targets, LDA_NONE_FILE)
    print(lda_none.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=True, applyPca=False)
    lda_ica = train_model(lda, data_features, data_targets, LDA_ICA_FILE)
    print(lda_ica.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=True)
    lda_pca = train_model(lda, data_features, data_targets, LDA_PCA_FILE)
    print(lda_pca.get_results())

    data_features, data_targets = featureExtraction.get_ICA_PCA_Data(applyIca=False, applyPca=False)
    lda_csp = train_model(lda, data_features, data_targets, LDA_CSP_FILE)
    print(lda_csp.get_results())


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
    print(svm_none.get_results())

    svm_ica = ModelSaver.read_model(SVM_ICA_FILE)
    print(svm_ica.get_results())

    svm_pca = ModelSaver.read_model(SVM_PCA_FILE)
    print(svm_pca.get_results())

    svm_csp = ModelSaver.read_model(SVM_CSP_FILE)
    print(svm_csp.get_results())

    lda_none = ModelSaver.read_model(LDA_NONE_FILE)
    print(lda_none.get_results())

    lda_ica = ModelSaver.read_model(LDA_ICA_FILE)
    print(lda_ica.get_results())

    lda_pca = ModelSaver.read_model(LDA_PCA_FILE)
    print(lda_pca.get_results())

    lda_csp = ModelSaver.read_model(LDA_CSP_FILE)
    print(lda_csp.get_results())


if retrain_models:
    train_models()
else:
    read_models()
