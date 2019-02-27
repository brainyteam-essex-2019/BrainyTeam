import os
import re
import numpy as np

# Reads Preprocessed data files
# from specified root folder with
# structure defined in documentation
def ReadData(root_dir = './preprocessed/', num_channels=64, ms = 300, fs = 240, transpose_runs=True):
    example_pattern = re.compile("example_[0-9]+[.]csv")
    num_samples = (int)(fs * (ms / 1000))

    data_dirs  = np.array(os.listdir(root_dir))
    data_files = np.array(os.listdir(root_dir))

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
                if transpose_runs:
                    run_examples[it] = np.transpose(file_data)
                it += 1
            else:
                run_targets = file_data

        examples = np.concatenate((examples, run_examples))
        targets = np.concatenate((targets, run_targets))

    return examples, targets