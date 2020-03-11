import numpy as np


def db2mag(db):
    mag = np.power(10., db / 20.)
    return mag

def mag2db(mag):
    db = 20 * np.log10(np.abs(mag)+np.finfo(float).eps)
    return db



def freq2time(fft_inp, norm_b=True):
    nb_frq = fft_inp.shape[0]
    if np.mod(nb_frq, 2) == 0:  # Even
        fft_inp = np.concatenate((fft_inp, np.conjugate(np.flipud(fft_inp[1:nb_frq]))), axis=0)
    else:
        fft_inp = np.concatenate((fft_inp, np.conjugate(np.flipud(fft_inp[1:nb_frq - 1]))), axis=0)
    # GAIN NORMALIZATION
    if norm_b:
        norm = 'ortho'
    else:
        norm = None
    # IFFT
    time_out = np.real(np.fft.ifft(fft_inp, axis=0, norm=norm))
    return time_out

def time2freq(frm_inp, nb_fft=None, norm_b=True):
    if nb_fft is None:
        nb_fft = frm_inp.shape[0]
    if np.mod(nb_fft, 2) == 0:  # Even
        nb_frq = nb_fft // 2 + 1
    else:
        nb_frq = nb_fft // 2
    # GAIN NORMALIZATION
    if norm_b:
        norm = 'ortho'
    else:
        norm = None
    # FFT
    fft_out = np.fft.fft(frm_inp, nb_fft, axis=0, norm=norm)[:nb_frq]
    return fft_out
