# -*- coding: utf-8 -*-
"""Crystallization demo model

This module contains a demo model for a simple crystallization

"""
# Importing dependencies
import numpy as np
from domain.Domain import Domain
from reference_model.Sample.ReferenceKinetics import Kinetics, Nucleation, Growth, Shrinkage
from reference_model.Sample.CrystallizationProperties import Solute
from reference_model.Sample.ReferenceModel import ReferenceModel


class CrystallizationDemoModel:
    def __init__(self, domain: Domain, save_intermediates=False):
        # Create domain
        self.domain = domain

        # Create kinetic sub-models
        self.nucleation_model = Nucleation(kp=0.5*10**2, p=3, ks=4.46*10**(-9), s=1.78)
        self.growth_model = Growth(kg=15/30, g=1.32, gamma=5*10**(-4))
        self.shrinkage_model = Shrinkage(ks=10/30, s=1, gamma=0)

        # Specify solubility parameters
        self.solute = Solute(solubility_parameters=[0.1286, -5.88*10**(-3), 1.721*10**(-4)],
                             crystal_density=2.1*10**(-9),
                             crystal_shape_factor=np.pi/6)

        # Create kinetic model
        self.kinetic_model = Kinetics(nucleation=self.nucleation_model,
                                      growth=self.growth_model,
                                      shrinkage=self.shrinkage_model)

        # Create crystallization model
        self.model = ReferenceModel(domain=self.domain,
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

    def start_new_batch(self, C0, T0, step_size, noise_level=1):
        # Initial conditions
        self.N0 = np.zeros(self.domain.axis[0].m)
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
        print('New batch started')
        measurement_object = self.model.measurement_object(np.expand_dims(self.t0_sim, axis=1),
                                                           np.expand_dims(self.x0, axis=1),
                                                           noise_level=self.noise_level)
        return measurement_object

    def get_next_measurement(self, z):
        if self.initialized:
            # Run simulation
            self.output = self.model.solve_ode(t0=self.t0_sim, timestep=self.step_size, x0=self.x_sim, z=[z])
            # Set new simulation parameters
            self.x_sim = self.output.y[:, -1]
            self.t0_sim = self.output.t[-1]
            # Save simulation output
            if self.save_intermediates:
                self.x_col.append(self.output.y)
                self.t_col.append(self.output.t)
            print(self.output.message)
            measurement_object = self.model.measurement_object(self.output.t, self.output.y,
                                                               noise_level=self.noise_level)
            return measurement_object
        else:
            print('Batch not initialized!')
