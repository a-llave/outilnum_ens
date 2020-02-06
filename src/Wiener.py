"""
Description:    wiener filter

Author: Martin Witt - ENS Rennes
Date: 30/01/2020

Version: 1

Date    | Auth. | Vers.  |  Comments
30/01/20   MWi     1.0       Initialization
06/02/20   MWi     2.0       Complete attenue

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import utils_llave as u

class Wiener:
    def __init__(self, nb_fft, RATE, win_size, delta_n=0.16, delta_s=0.09):
        # PARAMETERS
        if np.mod(nb_fft, 2) == 0:  # Even
            nb_frq = nb_fft // 2 + 1
        else:
            nb_frq = nb_fft // 2

        self.N_n = int(delta_n * 2 * RATE / win_size)

        self.N_s = int(delta_s * 2 * RATE / win_size)

        self.s_m = np.zeros((nb_frq, self.N_s), complex)
        self.n_m = np.zeros((nb_frq, self.N_n), complex)
        
    def process(self, x_v, s_v, n_v):
        # estimation
        self.s_m = np.concatenate((self.s_m[:, 1:], s_v), axis=1)
        self.n_m = np.concatenate((self.n_m[:, 1:], n_v), axis=1)
        
        sigma_s2_v = np.mean(np.abs(self.s_m)**2, axis=1)
        sigma_n2_v = np.mean(np.abs(self.n_m)**2, axis=1)

        # filter
        w_v = sigma_s2_v / (sigma_n2_v + sigma_s2_v)
        w_v = sigma_s2_v / (sigma_n2_v + sigma_s2_v)

        # w_v[w_v < u.db2mag(-40)] = u.db2mag(-40)
        # plt.clf()
        # plt.plot(sigma_s2_v)
        # plt.plot(sigma_n2_v)
        # plt.draw()
        # plt.pause(0.001)
        # plt.clf()
        # plt.plot(w_v)
        # plt.draw()
        # plt.pause(0.001)

        return w_v * x_v

