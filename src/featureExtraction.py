import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler


# concat the 3d set into 2d array
def concatenateData(whole):
    concatData = whole[0]
    for e in range(1, whole.shape[0]):
        concatData = np.append(concatData, whole[e], axis=0)
    return concatData


# get custom Data Set from examples
# if PCA set to true, does PCA transformation
# returns 3d array
def getAnalysisData(experiments, subjects, pca=False, p_componets=10):
    data = []
    targets = np.array([])
    pca = PCA(n_components=p_componets)
    sc = StandardScaler()

    for sub in range(subjects):
        for exp in range(1, experiments + 1):
            examples = np.array(os.listdir('./preprocessed/AAS01' + str(sub) + 'R0' + str(exp)))
            target = pd.read_csv('preprocessed/AAS01' + str(sub) + 'R0' + str(exp) + '/targets.csv', sep=",",
                                 header=None)
            targets = np.append(targets, target)
            for exa in examples:
                if exa != 'targets.csv':
                    current = pd.read_csv('preprocessed/AAS01' + str(sub) + 'R0' + str(exp) + '/' + str(exa), sep=",",
                                          header=None)
                    if pca == True:
                        current_std = sc.fit_transform(current)
                        current = pca.fit_transform(current_std)
                    data.append(np.array(current))
    data = np.asarray(data)
    print(data.shape)
    return data, targets


# produce a plot of Principal components Analysis
def analizePCA():
    data, targets = getAnalysisData(5, 2)
    concatData = concatenateData(data)
    pca = PCA()
    channels = pca.fit_transform(concatData)
    plt.bar(range(1, 65), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 65), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig('pca_plot.png', dpi=300)
    plt.show()


# produce a plot of Independent component Analysis
def analizeICA():
    data, targets = getAnalysisData(1, 1)
    concatData = concatenateData(data)
    ica = FastICA()
    S_ = ica.fit_transform(concatData)
    models = [concatData, S_]
    names = ['channels', 'ica']
    colors = ['red', 'steelblue', 'orange']
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig)
    plt.savefig('pci_plot.png', dpi=300)
    plt.show()

#Return the 3d data array as in UTIL applying PCA or ICA before
def get_ICA_PCA_Data(root_dir = './preprocessed/', num_channels=64, ms = 300, fs = 240, transpose_runs=True, applyPca = False, applyIca = False, p_components = 10, i_components = 0):
    if i_components != 0  :
        ica = FastICA(n_components = i_components)
    else :
        ica = FastICA()

    if applyPca == True :
        num_channels = p_components

    pca = PCA(n_components=p_components)
    sc = StandardScaler()

    example_pattern = re.compile("example_[0-9]+[.]csv")
    num_samples = (int)(fs * (ms / 1000))
    data_dirs  = np.array(os.listdir(root_dir))
    data_files = np.array(os.listdir(root_dir))

    print('chas', num_channels)
    if transpose_runs:
        examples = np.zeros((0, num_channels, num_samples))
    else:
        examples = np.zeros((0, num_samples, num_channels))
    targets = np.zeros((0))

    for run in data_dirs:
        data_files = np.array(os.listdir(root_dir + run + '/'))
        print(data_files.size)

        # MNE's CSP method requires data to be shaped num_channels x num_samples.
        if transpose_runs:
            run_examples = np.zeros((data_files.size - 1, num_channels, num_samples))
        else:
            run_examples = np.zeros((data_files.size - 1, num_samples, num_channels))

        it = 0
        for file in data_files:
            file_data = np.loadtxt(root_dir + run + '/' + file, delimiter=',')
            if (example_pattern.match(file)):
                data = file_data
                #ICA - PCA
                if applyPca == True :
                    std_examples = sc.fit_transform(data)
                    data = pca.fit_transform(std_examples)
                if applyIca == True :
                    data = ica.fit_transform(data)
                if transpose_runs:
                    run_examples[it] = np.transpose(data)
                else :
                    run_examples[it] = data
                it += 1
            else:
                run_targets = file_data
        examples = np.concatenate((examples, run_examples))
        targets = np.concatenate((targets, run_targets))

    return examples, targets

if __name__ == '__main__':
    # data, targets = getAnalysisData(1,1,True,11)
    data, targets = get_ICA_PCA_Data(applyPca = False,applyIca = True, p_components = 11)
