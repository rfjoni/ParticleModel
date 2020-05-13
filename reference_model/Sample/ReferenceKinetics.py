# -*- coding: utf-8 -*-
"""Reference kinetics module

This module includes reference kinetic from Nagy et al.

"""
# Importing dependencies
import numpy as np


class Kinetics:
    def __init__(self, nucleation=None, growth=None, shrinkage=None, agglomeration=None, breakage=None):
        self.models = dict()
        self.models['Nucleation'] = nucleation
        self.models['Growth'] = growth
        self.models['Shrinkage'] = shrinkage
        self.models['Agglomeration'] = agglomeration
        self.models['Breakage'] = breakage

    def rates(self, sigma, lengths, vc):
        rates = dict()
        for model_name, model in self.models.items():
            if model is not None:
                rates[model_name] = model.rate(sigma, lengths, vc)
            else:
                rates[model_name] = None
        return rates


class Nucleation:
    def __init__(self, kp, p, ks, s):
        self.kp = kp
        self.p = p
        self.ks = ks
        self.s = s

    def rate(self, sigma, lengths, vc):
        if sigma > 0:
            primary_rate = self.kp*sigma**self.p
            secondary_rate = self.ks*sigma**self.s*vc
        else:
            primary_rate = 0
            secondary_rate = 0
        return primary_rate+secondary_rate


class Growth:
    def __init__(self, kg, g, gamma):
        self.kg = kg
        self.g = g
        self.gamma = gamma

    def rate(self, sigma, lengths, vc):
        if sigma > 0:
            primary_rate = self.kg*sigma**self.g*(1+self.gamma*lengths)
        else:
            primary_rate = np.zeros_like(lengths)
        return primary_rate


class Shrinkage:
    def __init__(self, ks, s, gamma):
        self.ks = ks
        self.s = s
        self.gamma = gamma

    def rate(self, sigma, lengths, vc):
        if sigma < 0:
            primary_rate = self.ks*(1-sigma)**self.s*(1+self.gamma*lengths)
        else:
            primary_rate = np.zeros_like(lengths)
        return primary_rate


class Agglomeration:
    pass


class Breakage:
    pass
