# -*- coding: utf-8 -*-
"""Crystallization demo model

This module contains a demo model for a lactose crystallization

"""
# Importing dependencies
import numpy as np
from domain.Domain import Domain
from reference_model.Lactose.LactoseKinetics import LactoseSolute, LactoseNucleation, LactoseGrowth, LactoseKinetics
from reference_model.Lactose.LactoseModel import LactoseModel


class LactoseCrystallizer:
    def __init__(self, domain: Domain, save_intermediates=False, verbose=True):
        # Create domain
        self.domain = domain

        # Create kinetic sub-models
        self.nucleation_model = LactoseNucleation()
        self.growth_model = LactoseGrowth()

        # Specify solubility parameters
        self.solute = LactoseSolute()

        # Create kinetic model
        self.kinetic_model = LactoseKinetics(nucleation=self.nucleation_model,
                                             growth=self.growth_model)

        # Create crystallization model
        self.model = LactoseModel(domain=self.domain,
                                  kinetics=self.kinetic_model,
                                  solute=self.solute)

        # Initialized?
        self.initialized = False
        self.output = None
        self.N0 = None
        self.x0 = None
        self.x_sim = None
        self.t0_sim = None
        self.x_col = None
        self.t_col = None
        self.noise_level = None
        self.step_size = None
        self.save_intermediates = save_intermediates
        self.verbose = verbose

    def start_new_batch(self, N0, C0, T0, step_size, noise_level=1):
        # Initial conditions
        self.N0 = N0
        self.x0 = np.concatenate((self.N0, [C0], [T0]))
        self.step_size = step_size
        # Set update-able simulation results
        self.x_sim = self.x0
        self.t0_sim = 0
        self.x_col = []
        self.t_col = []
        self.output = None
        self.noise_level = noise_level
        self.initialized = True
        if not self.verbose:
            print('New batch started')
        measurement_object = self.model.measurement_object(self.t0_sim,
                                                           self.x0,
                                                           noise_level=self.noise_level)
        return measurement_object

    def get_next_measurement(self, z):
        if self.initialized:
            # Run simulation
            self.output = self.model.solve_ode(t0=self.t0_sim, timestep=self.step_size, x0=self.x_sim, z=[z])
            print(self.output)
            # Set new simulation parameters
            self.x_sim = self.output.y[:, -1]
            self.t0_sim = self.output.t[-1]
            # Save simulation output
            if self.save_intermediates:
                self.x_col.append(self.output.y)
                self.t_col.append(self.output.t)
            if not self.verbose:
                print(self.output.message)
            measurement_object = self.model.measurement_object(self.output.t, self.output.y,
                                                               noise_level=self.noise_level)
            return measurement_object
        else:
            if not self.verbose:
                print('Batch not initialized!')
