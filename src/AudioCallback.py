"""
Description: Class loading a audio matrix (nb_sample x nb_channels) to read it frame-by-frame

Author: Adrien Llave - CentraleSupelec
Date: 07/05/2019

Version: 1.3

Date    | Auth. | Vers.  |  Comments
28/03/18  ALl     1.0       First versionning
19/06/18  ALl     1.1       Add management of the end ('end_b' flag attribute, fill by zeros last buffer)
04/03/19  ALl     1.2       Add reset() method
07/05/19  ALl     1.3       Minor fix about the 'end_b' flag, make it rise a frame before

"""
import numpy as np


class AudioCallback:
    def __init__(self, sig_m, nb_buffsamp):
        if len(sig_m.shape) == 1:
            sig_m = sig_m[:, np.newaxis]
        elif len(sig_m.shape) > 2:
            print('FIRST INPUT ARGUMENT MUST BE A VECTOR OR A MATRIX')
        self.sig_m = sig_m
        self.nb_samples = self.sig_m.shape[0]
        self.nb_channels = self.sig_m.shape[1]
        self.nb_buffsamp = nb_buffsamp
        self.marker_in = 0
        self.end_b = False
        return

    def readframes(self):
        if self.marker_in+self.nb_buffsamp <= self.nb_samples:
            buffer_m = self.sig_m[self.marker_in:self.marker_in+self.nb_buffsamp, :]
        elif self.marker_in+self.nb_buffsamp > self.nb_samples and self.marker_in < self.nb_samples:
            buffer_m = np.concatenate((self.sig_m[self.marker_in:, :], np.zeros((self.marker_in+self.nb_buffsamp-self.nb_samples, self.nb_channels))))
            self.end_b = True
        else:
            buffer_m = np.zeros((self.nb_buffsamp, self.nb_channels))
        self.marker_in += self.nb_buffsamp
        return buffer_m

    def reset(self):
        self.marker_in = 0
        self.end_b = False
        return

