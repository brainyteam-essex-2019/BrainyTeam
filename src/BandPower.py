import numpy as np
import os
from scipy.signal import butter, lfilter, freqz
from scipy.signal import welch
from scipy.integrate import simps

# The setting signal Frequency
fs=240
# Define the duration of the window to be 4 seconds
win_sec=4


def bandpower(data, fs, band, window_sec=None, relative=False):
    
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * fs
    else:
        nperseg = (2 / low) * fs

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, fs, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


# get the lowpass_cutoff and highpass_cutoff per one signal.csv file
def gethighest_bp_cutoff(data,fs):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array  "data here is the signal"
        Input signal in the time-domain.
    fs : float
        Sampling frequency of the data.
    Return
    ------
    lowpass_cutoffall : 1d-array
            The lowpass_cutoff array, for example: lowpass_cutoff[0] is the first column lowpass_cutoff value
    highpass_cutoffall : 1d-array
            The highpass_cutoff array, for example: highpass_cutoff[0] is the first column highpass_cutoff value
    """

    Delta=[]
    Theta=[]
    Alpha=[]
    Beta=[]
    Gamma=[]
    Other=[]
    all_powerband=[]
    j=64
    lowpass_cutoffall=[]
    highpass_cutoffall=[]
    for i in range(j):
        data=signal[:,i]   #data here is the every column of the signal.csv file
        Delta_bp = bandpower(data, fs, [0.5, 4], win_sec,True)
        Theta_bp = bandpower(data, fs, [4, 8], win_sec,True) 
        Alpha_bp = bandpower(data, fs, [8, 12], win_sec,True) 
        Beta_bp = bandpower(data, fs, [12, 30], win_sec,True) 
        Gamma_bp = bandpower(data, fs, [30, 100], win_sec,True) 
        Other_bp = bandpower(data, fs, [100, 240], win_sec,True)
        Delta.append(Delta_bp)
        Theta.append(Theta_bp)
        Alpha.append(Alpha_bp)
        Beta.append(Beta_bp)
        Gamma.append(Gamma_bp)
        Other.append(Other_bp) 
    all_powerband.append(Delta)
    all_powerband.append(Theta)
    all_powerband.append(Alpha)
    all_powerband.append(Beta)
    all_powerband.append(Gamma)
    all_powerband.append(Other)
    for i in range(j):
        a=np.where(all_powerband[:,i]==np.max(all_powerband[:,i]))
        lowpass_cutoffall=[]
        highpass_cutoffall=[]
        if a[0][0]==0:
            lowpass_cutoff=0.5
            highpass_cutoff=4
        elif a[0][0]==1:
            lowpass_cutoff=4
            highpass_cutoff=8
        elif a[0][0]==2:
            lowpass_cutoff=8
            highpass_cutoff=12
        elif a[0][0]==3:
            lowpass_cutoff=12
            highpass_cutoff=30
        elif a[0][0]==4:
            lowpass_cutoff=30
            highpass_cutoff=100
        elif a[0][0]==5:
            lowpass_cutoff=100
            highpass_cutoff=240
        lowpass_cutoffall.append(lowpass_cutoff)
        highpass_cutoffall.append(highpass_cutoff)
    lowpass_cutoffall=np.array(lowpass_cutoffall)
    highpass_cutoffall=np.array(highpass_cutoffall)

    return lowpass_cutoffall,highpass_cutoffall



