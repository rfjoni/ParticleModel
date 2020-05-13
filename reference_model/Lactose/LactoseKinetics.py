# -*- coding: utf-8 -*-
"""Reference kinetics module

This module includes reference kinetics for lactose cooling crystallization

"""
# Importing dependencies
import numpy as np


class LactoseSolute:
    def __init__(self):
        self.solubility_parameters = [12.23, 0.3375, 0.001236, 0.00007257, 5.188*10**(-7)] # Visser (1982)
        self.crystal_density = 	1.525 # g/cm3 Preedy (2012)
        self.crystal_shape_factor = 1 # Own
        self.solubility_depression_parameters = [-2374.6, 4.5683] # Visser (1982)
        self.mutarotation_parameters = [-0.002286, 2.6371] # Visser (1982)


    def solubility(self, T):
        solubility = self.solubility_parameters[4]*T**4+\
                     self.solubility_parameters[3]*T**3+\
                     self.solubility_parameters[2]*T**2+\
                     self.solubility_parameters[1]*T+\
                     self.solubility_parameters[0]
        return solubility/100/1000 #[g anhydrous lactose/mL water]

    def solubility_depression(self, T):
        Tk = T+273.15
        F = np.exp(self.solubility_depression_parameters[0]/Tk+self.solubility_depression_parameters[1])
        return F

    def mutarotation(self, T):
        Km = self.mutarotation_parameters[0]*T+self.mutarotation_parameters[1]
        return Km

    def supersaturation(self, C, T):
        Cs = self.solubility(T)
        F = self.solubility_depression(T)
        Km = self.mutarotation(T)
        S = C / (Cs-F*Km*(C-Cs))
        return S

    def mass(self, N, domain):
        mass = np.sum(self.crystal_shape_factor * self.crystal_density * N * domain.axis[
            0].midpoints() ** 3 * 10 ** (-12))
        return mass


class LactoseKinetics:
    def __init__(self, nucleation=None, growth=None):
        self.models = dict()
        self.models['Nucleation'] = nucleation
        self.models['Growth'] = growth

    def rates(self, S, T, mc):
        rates = dict()
        for model_name, model in self.models.items():
            if model is not None:
                rates[model_name] = model.rate(S, T, mc)
            else:
                rates[model_name] = None
        return rates


class LactoseNucleation:
    def __init__(self):
        self.R = 8.314
        self.ks = 1.2*10**18/258/10**6/60/10**3 # [#/mL]
        self.Ea = -11.4
        self.s = 1.5

    def rate(self, S, T, mc):
        if S > 1:
            secondary_rate = self.ks*np.exp(self.Ea/(self.R*(T+273)))*np.abs(S-1)**self.s*mc*1000 # [#/(µL s)]
        else:
            secondary_rate = 0
        return secondary_rate


class LactoseGrowth:
    def __init__(self):
        self.R = 8.314
        self.kg = 3.2*10**9/39/60 #[µm]
        self.Ea = -22
        self.g = 2.4

    def rate(self, S, T, mc):
        if S > 1:
            primary_rate = self.kg*np.exp(self.Ea/(self.R*(T+273)))*np.abs(S-1)**self.g # [µm/s]
        else:
            primary_rate = 0
        return primary_rate