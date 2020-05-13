# -*- coding: utf-8 -*-
"""Crystallization specific properties

This module include crystallization specific submodels

"""


class Solute:
    def __init__(self, solubility_parameters, crystal_density, crystal_shape_factor):
        self.solubility_parameters = solubility_parameters
        self.crystal_density = crystal_density
        self.crystal_shape_factor = crystal_shape_factor

    def solubility(self, T):
        solubility = self.solubility_parameters[2]*T**2+self.solubility_parameters[1]*T+self.solubility_parameters[0]
        return solubility

    def supersaturation(self, C, T):
        solubility = self.solubility(T)
        supersaturation = (C-solubility) / solubility
        return supersaturation