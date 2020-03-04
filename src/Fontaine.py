# encoding: utf-8
"""
Description: Class Fontaine for ENS mini project

Author: Adrien Llave - CentraleSupelec
Date: 04/03/2020

Version: 2.1

Date    | Auth. | Vers.  |  Comments
30/01/20  VPe     1.0       Initialization
06/02/20  VPe     2.0       added class Fontaine débruitage un peu
04/03/20  ALl     2.1       - Bug fix: wrong way to implement the product between filter and input signal
                            - Ensure not doint log(0) in scale factor estimation

"""

import numpy as np
import sys
import scipy.stats as scistat


class Fontaine:
        def __init__(self, nb_fft, RATE, win_size, delta_n=0.16, delta_s=0.09, alpha_n=1.8, alpha_s=1.2):
            # PARAMETERS
            if np.mod(nb_fft, 2) == 0:  # Even
                nb_frq = nb_fft // 2 + 1
            else:
                nb_frq = nb_fft // 2

            self.alpha_s = alpha_s
            self.alpha_n = alpha_n

            self.gamma = np.euler_gamma  # constante d'Euler

            self.N_n = int(delta_n * 2 * RATE / win_size)

            self.N_s = int(delta_s * 2 * RATE / win_size)

            self.s_m = np.zeros((nb_frq, self.N_s), complex)
            self.n_m = np.zeros((nb_frq, self.N_n), complex)
            self.eps_f = 10 ** -8

        def process(self, x_v, s_v, n_v):
            # estimation
            self.s_m = np.concatenate((self.s_m[:, 1:], s_v), axis=1)
            self.n_m = np.concatenate((self.n_m[:, 1:], n_v), axis=1)

            lambda_s_v = np.exp((1 / self.alpha_s) * np.mean(np.log(np.abs(self.s_m)+self.eps_f), axis=1) + self.gamma * (1 - 1 / self.alpha_s))
            lambda_n_v = np.exp((1 / self.alpha_n) * np.mean(np.log(np.abs(self.n_m)+self.eps_f), axis=1) + self.gamma * (1 - 1 / self.alpha_n))

            phi_s_f = scistat.levy.median(loc=0, scale=2 * np.cos(np.pi * self.alpha_s / 4)**(2/self.alpha_s))
            phi_n_f = scistat.levy.median(loc=0, scale=2 * np.cos(np.pi * self.alpha_n / 4)**(2/self.alpha_n))

            # filter
            w_v = (phi_s_f * lambda_s_v**2) / (phi_s_f * lambda_s_v**2 + phi_n_f * lambda_n_v**2)

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
            s_estim_v = w_v[:, None] * x_v
            return s_estim_v

