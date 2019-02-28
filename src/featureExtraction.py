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


def analyzeCSP(examples, targets):

   csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

   csp_examples = csp.fit_transform(examples, targets)

   channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)

   bci_info = create_info(channel_names.tolist(), 240, ch_types='eeg', montage='biosemi64')
   csp.plot_patterns(bci_info, ch_type='eeg').savefig("CSP Patterns.png")
   csp.plot_filters(bci_info, ch_type='eeg').savefig("CSP Filters.png")

   return csp_examples, targets

# applys pca to the examples
# gets components numebr as parameter
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

# applys pca to the examples
# gets components numebr as parameter
def applyICA(components,examples,targets):
    tmin, tmax = -0.1, 0.3
    channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)
    epochs_info = create_info(channel_names[0:components].tolist(), 240, ch_types='eeg', montage='biosemi64')
    ica = UnsupervisedSpatialFilter(FastICA(components), average=False)
    ica_data = ica.fit_transform(examples)
    ev = mne.EvokedArray(np.mean(ica_data, axis=0),epochs_info, tmin=tmin)
    ev.plot(show=False, window_title="PCA", time_unit='s')
    plt.savefig('last_pci_plot.png', dpi=300)
    return examples,targets

if __name__ == '__main__':

    # data, targets = getAnalysisData(1,1,True,11)
    data, targets = ReadData()
    data,targets = applyPCA(30,data,targets)
