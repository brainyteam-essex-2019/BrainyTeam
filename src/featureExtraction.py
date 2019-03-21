# Authors : Zavala Jose, Tates Alberto
#
import pandas as pd
import numpy as np
import os
import re
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from mne.decoding import UnsupervisedSpatialFilter
from mne import create_info
from mne.decoding import CSP
from UTIL import ReadData

# Applies CSP to the examples, based on the number of parametrs as input 
# 
def analyzeCSP(components, examples, targets):

   csp = CSP(n_components=components, reg=None, log=True, norm_trace=False)

   csp_examples = csp.fit_transform(examples, targets)

   channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)

   bci_info = create_info(channel_names.tolist(), 240, ch_types='eeg', montage='biosemi64')
   csp.plot_patterns(bci_info, ch_type='eeg').savefig("CSP Patterns.png")
   csp.plot_filters(bci_info, ch_type='eeg').savefig("CSP Filters.png")

   return csp_examples, targets

# Applies PCA to the examples
# gets components numebr as parameter
#
def applyPCA(components,examples,targets):
    tmin, tmax = -0.1, 0.3
    channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)
    epochs_info = create_info(channel_names[0:components].tolist(), 240, ch_types='eeg', montage='biosemi64')
    pca = UnsupervisedSpatialFilter(PCA(components), average=False)
    pca_data = pca.fit_transform(examples)
    ev = mne.EvokedArray(np.mean(pca_data, axis=0),epochs_info, tmin=tmin)
    ev.plot(show=False, window_title="PCA", time_unit='s')
    plt.savefig('last_pca_plot.png', dpi=300)
    return examples,targets
   
# Produce a plot of Principal Component Analysis, the variance of each channel 
#
def analizePCA():
    data, targets = getData(5,2)
    concatData = data[0]
    for e in range(1,data.shape[0]):
        concatData = np.append(concatData,data[e],axis=0)
    data, targets = getPCAData(5,2)
    concatData = concatenateData(data)
    pca = PCA()
    channels = pca.fit_transform(concatData)

    plt.bar(range(1,65),pca.explained_variance_ratio_,alpha=0.5, align = 'center')
    plt.step(range(1,65),np.cumsum(pca.explained_variance_ratio_),where = 'mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig('pca_plot.png',dpi=300)
    plt.show()

# Produce a plot of Independent component Analysis, a comparison between the channels with and without ICA 
#
def analizeICA():
    data, targets = getPCAData(1,1)
    concatData = concatenateData(data)
    ica = FastICA()
    S_ = ica.fit_transform(concatData)
    models = [concatData,S_]
    names = ['channels', 'ica']
    colors = ['red', 'steelblue', 'orange']
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig)
    plt.savefig('pci_plot.png',dpi=300)
    plt.show()

# Applys pca to the examples
# gets components numebr as parameter
#
def applyICA(components,examples,targets):
    tmin, tmax = -0.1, 0.3
    channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)
    epochs_info = create_info(channel_names[0:components].tolist(), 240, ch_types='eeg', montage='biosemi64')
    ica = UnsupervisedSpatialFilter(FastICA(components), average=False)
    ica_data = ica.fit_transform(examples)
    ev = mne.EvokedArray(np.mean(ica_data, axis=0),epochs_info, tmin=tmin)
    ev.plot(show=False, window_title="PCA", time_unit='s')
    plt.savefig('last_ica_plot.png', dpi=300)
    return examples,targets

if __name__ == '__main__':

    # data, targets = getAnalysisData(1,1,True,11)
    data, targets = ReadData()
    data,targets = applyPCA(30,data,targets)
