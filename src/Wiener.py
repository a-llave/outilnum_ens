"""
Description:    wiener filter

Author: Adrien Llave - CentraleSupelec
Date: 04/03/2020

Version: 2.1

Date    | Auth. | Vers.  |  Comments
30/01/20   MWi     1.0       Initialization
06/02/20   MWi     2.0       Complete attenue
04/03/20  ALl       2.1      Bug fix: wrong way to implement the product between filter and input signal

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
        self.fig = None

    def process(self, x_v, s_v, n_v):
        # estimation
        self.s_m = np.concatenate((self.s_m[:, 1:], s_v), axis=1)
        self.n_m = np.concatenate((self.n_m[:, 1:], n_v), axis=1)
        
        sigma_s2_v = np.mean(np.abs(self.s_m)**2, axis=1)
        sigma_n2_v = np.mean(np.abs(self.n_m)**2, axis=1)

        # filter
        w_v = sigma_s2_v / (sigma_n2_v + sigma_s2_v)
        # nb_freq = sigma_s2_v.shape[0]
        # w_v = np.zeros(nb_freq)
        # for id_freq in range(nb_freq):
        #     w_v[id_freq] = sigma_s2_v[id_freq] / (sigma_n2_v[id_freq] + sigma_s2_v[id_freq])
        # self.w_v = self.coef * w_v + (1-self.coef) * self.w_v
        # PLOT
        # self.fig_update(sigma_s2_v, sigma_n2_v, w_v)
        s_estim_v = w_v[:, None] * x_v
        return s_estim_v

    def fig_init(self):
        self.fig = plt.figure()
        self.axs = [self.fig.add_subplot(1, 2, 1 + ii) for ii in range(2)]
        return

    def fig_update(self, sigma_s2_v, sigma_n2_v, w_v):
        if self.fig is None:
            self.fig_init()
        for ax in self.axs:
            ax.cla()
        w_v[w_v < u.db2mag(-40)] = u.db2mag(-40)
        self.axs[0].plot(sigma_s2_v, label='Source')
        self.axs[0].plot(sigma_n2_v, label='Noise')
        self.axs[0].legend()
        self.axs[1].plot(w_v)
        plt.draw()
        plt.pause(0.01)
        return
