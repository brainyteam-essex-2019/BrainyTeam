import numpy as np
import scipy.signal as sig
import math
'''This is a draft of phase locking value'''
'''need to comfirm if the format of a fixed window frequency domain is right'''
'''need to choose the best paramater j'''


#y1 and y2 are the comparable channels, they are 1-dimensional  matrix. j is the parameter, n is a period for each PLV 

def phase_locking(y1,y2,j,n):

    #calculate instantaneous phase value by Hilbert Transform
    y1_hilbert = sig.hilbert(y1)
    y2_hilbert = sig.hilbert(y2)
    y1_ipv = (np.arctan(y1_hilbert/y1))
    y2_ipv = (np.arctan(y2_hilbert/y2))
   
  

    #calculate PLV
    PLV = []
    for i in range(len(y1)):
        if i == 0:
            dif_plv = math.exp(j*(y1_ipv[i] - y2_ipv[i]))
        elif i%n != 0:
            dif_plv = math.exp(j*(y1_ipv[i] - y2_ipv[i])) + dif_plv
        else:
            plv = dif_plv/n
            dif_plv = math.exp(j*(y1_ipv[i] - y2_ipv[i]))
            PLV.append(plv)
    return PLV

