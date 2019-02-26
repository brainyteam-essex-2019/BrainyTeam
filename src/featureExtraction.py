import pandas as pd
import numpy as np
import os
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
def getPCAData(experiments, subjects, pca=False, p_componets=10):
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
    data, targets = getPCAData(5, 2)
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
    data, targets = getPCAData(1, 1)
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


# get custom Data Set from examples applying ICA
# n_components for ICA
# returns 3d array
def getICAData(experiments, subjects):
    data, targets = getPCAData(experiments, subjects)
    concatData = concatenateData(data)
    ica = FastICA()
    S_ = ica.fit_transform(concatData)
    return S_, targets


if __name__ == '__main__':
    # data, targets = getPCAData(1,1,True,11)
    data, targets = getICAData(1, 1)
    print(data)
