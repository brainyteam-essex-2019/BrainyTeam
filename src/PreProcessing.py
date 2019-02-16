import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz

# ------------- UTIL ------------- #

# Finds next flashing window start
def findNextExample(it, flashing):
  while (it < flashing.size):
    if (flashing[it]):
      return it
    it += 1
    
  return sys.maxsize
  
# Plots lowpass frequency response
def plotLowpass(cutoff, fs=240, order=5):
  b, a = butter(order, cutoff, btype='lowpass', analog=False, fs=fs)
  w, h = freqz(b, a, worN=8000)
  
  plt.subplot(2, 1, 1)
  plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
  plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
  plt.axvline(cutoff, color='k')
  plt.xlim(0, 0.1*fs)
  plt.title("Lowpass Filter Frequency Response")
  plt.xlabel('Frequency [Hz]')
  plt.grid()
  
  plt.show()
  
# Plots lowpass frequency response
def plotHighpass(cutoff, fs=240, order=5):
  b, a = butter(order, cutoff, btype='highpass', analog=False, fs=fs)
  w, h = freqz(b, a, worN=8000)
  
  plt.subplot(2, 1, 1)
  plt.plot((0.5*fs)*w/np.pi, np.abs(h), 'b')
  plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
  plt.axvline(cutoff, color='k')
  plt.xlim(0, 0.1*fs)
  plt.title("Highpass Filter Frequency Response")
  plt.xlabel('Frequency [Hz]')
  plt.grid()
  
  plt.show()
  
# ------------- PRE-PROCESSING ------------- #

# Reads signal, stimulus and flashing files for a given run
def readData(dirName):
  signal = np.genfromtxt(dirName + "signal.csv", delimiter=',', dtype=np.float64)
  
  stimulus_int = np.genfromtxt(dirName + "StimulusType.csv", delimiter=',', dtype=int)
  stimulus = stimulus_int == 1
  
  flashing_int = np.genfromtxt(dirName + "Flashing.csv", delimiter=',', dtype=int)
  flashing = flashing_int == 1
  
  return signal, stimulus, flashing

# Takes full matrix and applies lowpass filter to each brain signal
def lowpassFilter(signal, cutoff, fs=240, order=5):

  b, a = butter(order, cutoff, btype='lowpass', analog=False, fs=fs)

  y = lfilter(b, a, signal, axis=0)

  return y

# Takes full matrix and applies highpass filter to each brain signal
def highpassFilter(signal, cutoff, fs=240, order=5):

  b, a = butter(order, cutoff, btype='highpass', analog=False, fs=fs)

  y = lfilter(b, a, signal, axis=0)

  return y
  
  
# Creates Training Examples based on a number of samples after a stimulus is given
# ms is the number of milliseconds a training example will contain
# Returns:
#   3-D data matrix in which:
#     Dim-0 -> Number of training examples
#     Dim-1 -> Signals samples through time
#     Dim-2 -> Brain sensor which read the signal
#   1-D targets array in which:
#     Dim-0 -> Binary value that indicates if a P300 wave was fired.
def genExamplesFromSignal(signal, stimulus, flashing, fs=240, ms=300, flash_ms = 100):
  fls_size = (int)(fs * (flash_ms / 1000))
  wnd_size = (int)(fs * (ms / 1000))
  
  targets = np.array([]).reshape(0).astype(bool)
  
  it = findNextExample(0, flashing)
  
  # First example
  examples = (signal[it:it+wnd_size], )
  it = findNextExample(it + fls_size, flashing)
  
  while (it < stimulus.size):
    examples = examples + (signal[it:it+wnd_size], )
    targets = np.append(targets, stimulus[it])
    
    it = findNextExample(it + fls_size, flashing)
  
  data = np.stack(examples)
  
  return data, targets

# Save the pre-processed training examples.
def saveExamples(data, targets, dirName):
  try:
    os.makedirs(dirName)
  except FileExistsError:
      # directory already exists
      pass

  for it in range(data.shape[0]):
    np.savetxt(dirName + "example_" + str(it) + ".csv", data[it], delimiter=",")
    
  np.savetxt(dirName + "targets.csv", targets, delimiter=",")
  
  
if __name__ == '__main__':
  runNames = np.array(os.listdir('./data'))
  
  lowpass_cutoff = 10
  highpass_cutoff = 1
    
  plotLowpass(lowpass_cutoff)
  plotHighpass(highpass_cutoff)
  
  for run in runNames:
    dirName    = './data/'         + run + '/'
    resultName = './preprocessed/' + run + '/'
  
    signal, stimulus, flashing = readData(dirName)
    
    signal = lowpassFilter(signal, lowpass_cutoff)
    signal = highpassFilter(signal, highpass_cutoff)
    
    data, targets = genExamplesFromSignal(signal, stimulus, flashing)
    
    saveExamples(data, targets, resultName)
