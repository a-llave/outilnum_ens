# encoding: utf-8
"""
Description: Base for ENS mini project

Author: Adrien Llave - CentraleSupelec
Date: 29/01/2020

Version: 1.0

Date    | Auth. | Vers.  |  Comments
29/01/20   ALl     1.0       Initialization
06/02/20   MWi     2.0       Complete attenue

"""

import numpy as np
# import RTAudioProc as rt
from src.BlockProc import *
from src.AudioCallback import *
from src.Fontaine import *
import scipy.io.wavfile as wavfile
import scipy.stats as scistat
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.cm as cm
import src.utils_hack as u
import sounddevice as sd
import copy
from scipy.io import wavfile
import os
import sys
import mir_eval as me
import time

# ----------------------------------------------------------------
# noise_s = 'white'
noise_s = 'cafet'

# ---- PATH ------------------------------------------------------------
folder_stim = '../resources/samples'

# ---- GENERAL
RATE = 16000
win_size = 128  # 128
buf_size = win_size // 2
nb_fft = win_size
nb_freq = nb_fft // 2 + 1
duration = 4
nb_sample = duration * RATE
nb_buf = int(nb_sample / buf_size) - 1

# SPEECH
fs, src_v = wavfile.read(os.path.join(folder_stim, 'sample_1.wav'))
src_v = copy.deepcopy(src_v) / 2 ** 16 * u.db2mag(-24)
src_v = np.concatenate((np.zeros(RATE // 2, ), src_v[:-RATE // 2]))

# NOISE
if noise_s == 'white':
    nse_v = 0.001 * np.random.randn(src_v.shape[0])
elif noise_s == 'cafet':
    fs, nse_v = wavfile.read(os.path.join(folder_stim, 'cafet_16k_mono.wav'))
    nse_v = copy.deepcopy(nse_v) / 2 ** 16 * u.db2mag(-12)
    nse_v = nse_v[:src_v.shape[0]]

# MAKE MIX
stim_v = copy.copy(src_v)
stim_v += nse_v
stim = AudioCallback(stim_v, buf_size)
ntim = AudioCallback(nse_v, buf_size)
srctim = AudioCallback(src_v, buf_size)

# ====== DEFINE PROCESS ========
bloc_stim = BlockProc(nb_buffsamp=buf_size, nb_channels_inp=1, nb_channels_out=1)
bloc_src = BlockProc(nb_buffsamp=buf_size, nb_channels_inp=1, nb_channels_out=1)
bloc_nse = BlockProc(nb_buffsamp=buf_size, nb_channels_inp=1, nb_channels_out=1)

SIR = []
SDR = []
SAR = []

alph = np.linspace(0.8, 1.5, 10)
print(alph)
for aln in alph:
    fontaine = Fontaine(nb_fft, RATE, win_size, 0.16, 0.09, aln, 1.2)
    # ============= PROCESS ===============================
    src_est_v = np.zeros(stim_v.shape)
    stim.reset()
    fontaine.reset()
    for ii in range(nb_buf):
        bfr_inp = stim.readframes()
        frm_inp = bloc_stim.input2frame(bfr_inp)
        fft_inp = u.time2freq(frm_inp, nb_fft)

        # SRC
        bfr_src = srctim.readframes()
        frm_src = bloc_src.input2frame(bfr_src)
        fft_src = u.time2freq(frm_src, nb_fft)
        # noise
        bfr_nse = ntim.readframes()
        frm_nse = bloc_nse.input2frame(bfr_nse)
        fft_nse = u.time2freq(frm_nse, nb_fft)

        # TODO: INSERT THE PROCESSING HERE
        fft_out = fontaine.process(fft_inp, fft_src, fft_nse)
        # fft_out = fft_inp

        frm_out = u.freq2time(fft_out)
        bfr_out = bloc_stim.frame2output(frm_out)
        src_est_v[ii * buf_size:(ii + 1) * buf_size] = bfr_out[:, 0]

    src_est_v = src_est_v / np.max(np.abs(src_est_v)) * np.max(np.abs(src_v))

    # ======== RESULTS ===================
    ref_srcs = np.hstack((src_v[:, None], nse_v[:, None])).T
    est_srcs = np.hstack((src_est_v[:, None], nse_v[:, None])).T
    sdr, sir, sar, popt = me.separation.bss_eval_sources(ref_srcs, est_srcs)
    SIR.append(sir[0])
    SDR.append(sdr[0])
    SAR.append(sar[0])

plt.plot(alph, SIR)
plt.show()
plt.plot(alph, SAR)
plt.show()
plt.plot(alph, SDR)
plt.show()

# print('SDR:\n--- SRC: %.0f dB\n--- NSE: %.0f dB' % (sdr[0], sdr[1]))
# print('SIR:\n--- SRC: %.0f dB\n--- NSE: %.0f dB' % (sir[0], sir[1]))
# print('SAR:\n--- SRC: %.0f dB\n--- NSE: %.0f dB' % (sar[0], sar[1]))
#
# print('----------- PLAY -------------------------------')
# sd.play(stim_v, samplerate=RATE, blocking=True)
# sd.play(src_est_v, samplerate=RATE, blocking=True)
#
# print('------------- SAVE --------------------------------')
# wavfile.write(os.path.join('../resources', 'output', 'speech_noise.wav'), rate=RATE, data=stim_v)
# wavfile.write(os.path.join('../resources', 'output', 'speech_clean.wav'), rate=RATE, data=src_est_v)
