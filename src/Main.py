import featureExtraction
import PreProcessing
import ModelSaver
from GenSVM import SVMModel
from LDAClassifier import LDAModel

"""
Inputs needed
1: Retrain models or not 
2: use PCA or ICA
3: What Window size to use? 
4: Preprocessing or not 
"""

LDA_PICKLE_FILE = "LDA_MODEL.pkl"
SVM_PICKLE_FILE = "SVM_MODEL,pkl"

# variables set by user
pre_processing = True
retrain_models = True
use_ica = True
use_pca = False  # kept if add third feature extraction  method
window_size = 300
data_features = None
data_targets = None

# models
svm = SVMModel()
lda = LDAModel()

if pre_processing:
    print()
# run main in pre-processing

if use_ica:
    # data_features, data_targets = featureExtraction.getICAData()
    featureExtraction.analizeICA()
else:  # use PCA
    # data_features, data_targets = featureExtraction.getPCAData()
    featureExtraction.analizePCA()
if retrain_models:
    # train & save SVM model
    svm.load_data(data_features)
    svm.load_target(data_targets)
    svm.train()
    ModelSaver.save_model(svm, SVM_PICKLE_FILE)

    # train & save SVM model
    lda.load_data(data_features)
    lda.load_target(data_targets)
    lda.train()
    ModelSaver.save_model(lda, LDA_PICKLE_FILE)
else:
    lda = ModelSaver.read_model(LDA_PICKLE_FILE)
    svm = ModelSaver.read_model(SVM_PICKLE_FILE)