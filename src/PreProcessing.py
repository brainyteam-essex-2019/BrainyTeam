import numpy as np

from scipy.signal import butter, lfilter, freqz

# Reads signal file of a given run
def readSignal(fileName):
  signal = np.array([]).reshape((0, 0))
  # TODO
  return signal

# Reads stimulus file of a given run
def readStimulus(fileName):
  stimulus = np.array([]).reshape(0)
  # TODO:
  return stimulus

# Takes full matrix and applies lowpass filter to each brain signal
def lowpassFilter(signal, cutoff, fs=240, order=5):

  b, a = butter(order, cutoff, btype='lowpass', analog=False, fs=fs)

  y = lfilter(b, a, signal, axis=1)

  return y

# Takes full matrix and applies highpass filter to each brain signal
def highpassFilter(signal, cutoff, fs=240, order=5):

  b, a = butter(order, cutoff, btype='highpass', analog=False, fs=fs)

  y = lfilter(b, a, signal, axis=1)

  return y

# Creates Training Examples based on a number of samples after a stimulus is given
# ms is the number of milliseconds a training example will contain
# Returns:
#   3-D data matrix in which:
#     Dim-0 -> Number of training examples
#     Dim-1 -> Signals amples through time
#     Dim-2 -> Brain sensor which read the signal
#   1-D targets array in which:
#     Dim-0 -> Binary value that indicates if a P300 wave was fired.
def genExamplesFromSignal(signal, stimulus, fs=240, ms=300):
  data = np.array([]).reshape((0, 0, 0))
  targets = np.array([]).reshape(0)

  # TODO

  return data, targets

# Save the pre-processed training examples.
def saveExamples(signal, targets, fileName):
  # TODO

  return

