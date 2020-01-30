"""
Description:    Overlap add framework

Author: Adrien Llave - CentraleSupelec
Date: 20/01/2020

Version: 1.3

Date    | Auth. | Vers.  |  Comments
22/05/18  ALl     1.0       Initialization
06/03/19  ALl     1.1       MAJOR CHANGE:
                                - add analysis window
                                - Change management of the frame getter and setter
08/11/19  ALl     1.2       Add reset() method
20/01/20  ALl     1.3       Code factorization in update_window()

"""

import numpy as np
import sys


class BlockProc:
    def __init__(self,
                 nb_buffsamp,
                 nb_channels_inp,
                 nb_channels_out,
                 window_inp_s='sqrt-hann',
                 window_out_s='sqrt-hann',
                 gain_normalization_b=True):
        # PARAMETERS
        self.buffer_count = 0
        self.nb_buffsamp = nb_buffsamp
        self.nb_framesamp = self.nb_buffsamp * 2
        self.nb_channels_inp = nb_channels_inp
        self.nb_channels_out = nb_channels_out
        self.gain_normalization_b = gain_normalization_b
        # LINES
        self.buffer_inp_prev = np.zeros((self.nb_buffsamp, self.nb_channels_inp))
        self._frame0_m = np.zeros((self.nb_framesamp, self.nb_channels_out))
        self._frame1_m = np.zeros((self.nb_framesamp, self.nb_channels_out))
        # WINDOWS
        self.window_inp_s = window_inp_s
        self.window_out_s = window_out_s
        self.window_inp_m = np.zeros((self.nb_framesamp, self.nb_channels_inp))
        self.window_out_m = np.zeros((self.nb_framesamp, self.nb_channels_out))
        self.update_window()

    def input2frame(self, buffer_inp):
        frame_win = np.concatenate((self.buffer_inp_prev, buffer_inp), axis=0) * self.window_inp_m
        self.buffer_inp_prev = buffer_inp
        return frame_win

    def frame2output(self, frame_out):
        self.buffer_count = 1 - self.buffer_count

        if not self.buffer_count:
            self._frame0_m = frame_out * self.window_out_m
        else:
            self._frame1_m = frame_out * self.window_out_m

        if self.buffer_count:
            buf_out = self._frame0_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :] + \
                      self._frame1_m[0:self.nb_buffsamp, :]
        else:
            buf_out = self._frame0_m[0:self.nb_buffsamp, :] + \
                      self._frame1_m[self.nb_buffsamp:self.nb_buffsamp + self.nb_buffsamp, :]

        return buf_out

    def update_window(self, window_inp_s=None, window_out_s=None):
        """

        :param window_inp_s:
        :param window_out_s:
        :return:
        """
        if window_inp_s is not None:
            self.window_inp_s = window_inp_s
        if window_out_s is not None:
            self.window_out_s = window_out_s

        # UPDATE INPUT WINDOW
        if self.window_inp_s == 'blackman':
            window_v = np.blackman(self.nb_framesamp)
        elif self.window_inp_s == 'hann':
            window_v = np.hanning(self.nb_framesamp)
        elif self.window_inp_s == 'sqrt-hann':
            window_v = np.sqrt(np.hanning(self.nb_framesamp))
        elif self.window_inp_s == 'ones':
            window_v = np.ones((self.nb_framesamp,))
        else:
            print('BlockProc::update_window: Unknown window type.', file=sys.stderr)
            window_v = 0
        if self.gain_normalization_b:
            gain_norm_f = np.sqrt(np.mean(np.square(window_v)))
            window_v /= gain_norm_f
        self.window_inp_m = np.repeat(window_v[:, None], self.nb_channels_inp, axis=1)

        # UPDATE OUTPUT WINDOW
        if self.window_out_s == 'blackman':
            window_v = np.blackman(self.nb_framesamp)
        elif self.window_out_s == 'hann':
            window_v = np.hanning(self.nb_framesamp)
        elif self.window_out_s == 'sqrt-hann':
            window_v = np.sqrt(np.hanning(self.nb_framesamp))
        elif self.window_out_s == 'ones':
            window_v = np.ones((self.nb_framesamp,))
        else:
            print('BlockProc::update_window: Unknown window type.', file=sys.stderr)
            window_v = 0
        if self.gain_normalization_b:
            window_v *= gain_norm_f
        self.window_out_m = np.repeat(window_v[:, None], self.nb_channels_out, axis=1)

        return
    
    def reset(self):
        self.buffer_inp_prev = np.zeros((self.nb_buffsamp, self.nb_channels_inp))
        self._frame0_m = np.zeros((self.nb_framesamp, self.nb_channels_out))
        self._frame1_m = np.zeros((self.nb_framesamp, self.nb_channels_out))
        return
