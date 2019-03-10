import numpy as np
import scipy.signal as sig
import cmath
''' I don't know how to use a imaginary funcion exp() for line26'''

#y1 and y2 are the comparable channels, they are 1-dimensional  matrix.

def phase_locking(channel1,channel2):

    #calculate instantaneous phase value by Hilbert Transform
    '''There are two calculate the instantaneous phase value'''
    '''The first one according to the literature I found'''
    sig1_hill=sig.hilbert(channel1)
    sig2_hill=sig.hilbert(channel2)
    phase_y1 = np.arctan(sig1_hill/channel1)
    phase_y2 = np.arctan(sig2_hill/channel2)

    '''The second one is according the matlab function''' 
    #phase_y1=np.unwrap(np.angle(sig1_hill))
    #phase_y2=np.unwrap(np.angle(sig2_hill))

    #calculate phase locking value
    sum = 0
    for i in range(len(phase_y1)):
        dif = phase_y1[i] - phase_y2[i]
        variance = cmath.exp((dif))         
        sum += variance
    PLV = sum/len(phase_y1)
    
    return PLV
