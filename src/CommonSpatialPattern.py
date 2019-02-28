import numpy as np

from mne import create_info
from mne.decoding import CSP
from UTIL import ReadData

examples, targets = ReadData()

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

csp_examples = csp.fit_transform(examples, targets)

channel_names = np.loadtxt('./metadata/channel_names.csv', dtype=str)

bci_info = create_info(channel_names.tolist(), 240, ch_types='eeg', montage='biosemi64')
csp.plot_patterns(bci_info, ch_type='eeg')
csp.plot_filters(bci_info, ch_type='eeg')
