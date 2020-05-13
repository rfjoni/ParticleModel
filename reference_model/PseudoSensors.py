# -*- coding: utf-8 -*-
"""Pseudo sensor module

This module emulates real process sensors by transforming simulation data to data-objects with uncertainty noise

"""
# Importing dependencies
import itertools
import numpy as np


class PseudoSensors:
    def __init__(self, domain, noise_level=1):
        self.domain = domain
        self.noise_level = noise_level

    def image_analysis(self, N):
        # Generate pseudo image analysis result from size-distribution
        ia_data = list(itertools.chain.from_iterable(
            itertools.repeat(length, int(N[bin_index])) for bin_index, length in enumerate(self.domain.axis[0].midpoints())))
        np.random.shuffle(ia_data)
        return np.asarray(ia_data)

    def PSD_noise(self, N):
        N_noisy = np.zeros_like(N)
        N = np.abs(N)
        number_particles = np.max([1, np.sum(N)])
        for bin_id in range(len(N)):
            stdev = self.noise_level*np.sqrt(N[bin_id]*(1-N[bin_id]/number_particles))
            if stdev > 0:
                N_noisy[bin_id] = N[bin_id]+np.random.normal(loc=0, scale=stdev)
            else:
                N_noisy[bin_id] = round(N[bin_id])
        return N_noisy

    def sensor_noise(self, x, stdev):
        x_noise = x+np.random.normal(loc=np.zeros_like(x), scale=self.noise_level*stdev)
        return x_noise
