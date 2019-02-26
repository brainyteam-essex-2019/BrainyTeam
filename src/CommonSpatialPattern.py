import os
import re
import numpy as np

from mne import create_info
from mne.decoding import CSP


# READ DATA FOR AAS01R01
ROOT_DIR = "./preprocessed/AAS010R01/"
example_pattern = re.compile("example_[0-9]+[.]csv")
num_channels = 64
num_samples = 72

data_files = np.array(os.listdir(ROOT_DIR))

# MNE's CSP method requires data to be shaped num_channels x num_samples.
examples = np.zeros((data_files.size - 1, num_channels, num_samples))

it = 0
for file in data_files:
	file_data = np.loadtxt(ROOT_DIR + file, delimiter=',')
	if (example_pattern.match(file)):
		examples[it] = np.transpose(file_data)
		it += 1
	else:
		targets = file_data

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

csp_examples = csp.fit_transform(examples, targets)

channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)

bci_info = create_info(channel_names.tolist(), 240, ch_types='eeg', montage='biosemi64')
csp.plot_patterns(bci_info, ch_type='eeg')
csp.plot_filters(bci_info, ch_type='eeg')
