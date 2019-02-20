import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# get custom Data Set from examples
# if PCA set to true, does PCA transformation
#
# returns 3d array
def getData(experiments, subjects, pca = False, p_componets = 10):
    data = []
    targets = np.array([])
    pca = PCA(n_components = p_componets)
    sc = StandardScaler()

    for sub in range(subjects):
        for exp in range(1,experiments+1):
            examples = np.array(os.listdir('./preprocessed/AAS01'+str(sub)+'R0'+str(exp)))
            target = pd.read_csv('preprocessed/AAS01'+str(sub)+'R0'+str(exp)+'/targets.csv',sep = ",", header = None)
            targets = np.append(targets, target)
            for exa in examples:
                if exa != 'targets.csv':
                    current = pd.read_csv('preprocessed/AAS01'+str(sub)+'R0'+str(exp)+'/'+str(exa),sep = ",", header= None)
                    if pca == True:
                        current_std = sc.fit_transform(current)
                        current = pca.fit_transform(current_std)
                    data.append(np.array(current))
    data = np.asarray(data)
    print(data.shape)
    return data, targets

# produce a plot of P(rincipal components Analisis
def analizePCA():
    data, targets = getData(5,2)
    concatData = data[0]
    for e in range(1,data.shape[0]):
        concatData = np.append(concatData,data[e],axis=0)
    pca = PCA()
    channels = pca.fit_transform(concatData)

    plt.bar(range(1,65),pca.explained_variance_ratio_,alpha=0.5, align = 'center')
    plt.step(range(1,65),np.cumsum(pca.explained_variance_ratio_),where = 'mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig('pca_plot.png',dpi=300)
    plt.show()


if __name__ == '__main__':
    data, targets = getData(1,1,True,11)
    print(data)
