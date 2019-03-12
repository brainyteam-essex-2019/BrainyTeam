import numpy as np
import scipy.signal as sig
import cmath
import math
from numpy.lib.stride_tricks import as_strided


def padded_sliding_windows(a, split_size, pad_length, padnum):
    # n = a.strides[0]
    # L = split_size + pad_length
    # S = L - pad_length
    # nrows = ((a.size + pad_length - L)//split_size)+1
    b = np.pad(a, (pad_length, 0), mode='constant', constant_values=padnum)
    # return a memory efficient view on the array
    return as_strided(b,
                      shape=(b.size//split_size, split_size + pad_length),
                      strides=(b.strides[0]*split_size, b.strides[0]))


class PhaseLockProcessor:

    def __init__(self, step_length, step_padding):
        self.step_length = step_length
        self.step_padding = step_padding

    def calculate_unit(self, slice1, slice2):

        # Apply hilbert transformation to the signals
        sig1_hill = sig.hilbert(slice1)
        sig2_hill = sig.hilbert(slice2)

        d = np.sqrt(np.inner(sig1_hill,
                             np.conj(sig1_hill)) *
                    np.inner(sig2_hill,
                             np.conj(sig2_hill)))

        pdt = (np.inner(sig1_hill, np.conj(sig2_hill)) / d) if d != 0 else 0

        # calculate the $e^{j * \phi (t)}$
        pl = cmath.exp(1j * pdt)
        return pl

    # Assume that channel1 and channel2 are same length
    # Sample channel1 and channel2 with sliding time window of self.step_length and self.step_padding,
    # and calculate a single plv value aggreating the sliding time windowed signal
    def calculate_plv(self, channel1, channel2):
        channel1_sliding = padded_sliding_windows(
            channel1, split_size=self.step_length, pad_length=self.step_padding, padnum=0)
        channel2_sliding = padded_sliding_windows(
            channel2, split_size=self.step_length, pad_length=self.step_padding, padnum=0)
        out = 0
        for i in range(channel1_sliding.shape[0]):
            t = self.calculate_unit(
                channel1_sliding[i], channel2_sliding[i])
            out += t
        return np.abs(out) / channel1_sliding.shape[0]

    # signals should be in shape channels x signals
    # return in shape plv_steps x channels x channels, each plv_step stands for an observation of plv of a certain period of time
    def gen_channel_plv_matrix(self, chan_signals, observe_step_length, observe_step_padding):
        chan_signals_sliding = [padded_sliding_windows(
            ch, split_size=observe_step_length, pad_length=observe_step_padding, padnum=0)
            for ch in chan_signals]
        out = np.zeros(
            (chan_signals_sliding.shape[1], chan_signals_sliding.shape[0], chan_signals_sliding.shape[0]))
        for sliding_idx in range(chan_signals_sliding.shape[1]):
            for ch1_idx in range(chan_signals_sliding.shape[0]):
                for ch2_idx in range(chan_signals_sliding.shape[0]):
                    out[sliding_idx, ch1_idx, ch2_idx] = self.calculate_plv(
                        chan_signals_sliding[ch1_idx, sliding_idx],
                        chan_signals_sliding[ch2_idx, sliding_idx],
                    )
        return out
