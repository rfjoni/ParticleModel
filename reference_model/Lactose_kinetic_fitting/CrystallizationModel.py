# -*- coding: utf-8 -*-
"""Reference model module

This module contains code for reference model for lactose crystallization

"""
# Importing dependencies
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime
from data.Data import Measurement, Sensor
from reference_model.PseudoSensors import PseudoSensors


class CrystallizationModel:
    def __init__(self, domain, kinetics, solute):
        self.domain = domain
        self.kinetics = kinetics
        self.solute = solute

    def rates(self, kinetics, N):
        # Rates dictionary
        bin_rates = dict()
        # Calculate dNdt
        for kinetic_name, rate in kinetics.items():
            if rate is not None:
                # Initialize rate
                birth_rate = np.zeros([self.domain.axis[0].m])
                death_rate = np.zeros([self.domain.axis[0].m])
                # For each type of phenomena:
                if kinetic_name == 'Nucleation':
                    # Set birth and death rate
                    birth_rate[0] = rate
                elif kinetic_name == 'Growth':
                    # Set birth and death rate
                    gp = N*rate/(2*self.domain.axis[0].widths())
                    birth_rate = np.concatenate([[0], gp[:-1]])
                    death_rate = gp
                bin_rates[kinetic_name] = birth_rate - death_rate
        return bin_rates

    def dxdt(self, x, t, z, parameters):
        # Initialize dNdt and dTdt
        dNdt = np.zeros([self.domain.axis[0].m])
        T = np.polyval(z, t)

        # Split x to N, C and T
        N = x[:self.domain.axis[0].m]
        C = x[-1]
        # Calculate sigma and total volume
        S = self.solute.supersaturation(C=C, T=T, parameters=parameters)
        mc = self.solute.mass(N=N, domain=self.domain, parameters=parameters) # g lactose/µL

        # Calculate phenomena kinetics
        kinetics = self.kinetics.rates(S=S, mc=mc, T=T, parameters=parameters)

        # Rates
        bin_rates = self.rates(kinetics=kinetics, N=N)

        # Calculate dNdt
        for kinetic_name, bin_rate in bin_rates.items():
            dNdt = dNdt + bin_rate

        # Calculate dCdt
        dCdt = -self.solute.mass(N=dNdt, domain=self.domain, parameters=parameters)

        # Collect as dxdt
        dxdt = np.concatenate([dNdt, [dCdt]])
        return dxdt

    def solve_ode(self, t_steps, z, parameters, x0):
        # Solve ode
        z = solve_ivp(fun = lambda t, x: self.dxdt(x, t, z, parameters),
                      y0=x0, t_span=(t_steps[0], t_steps[-1]),
                      method='LSODA', t_eval=t_steps)
        return z

    def loss(self, t_ref, N_ref, z, parameters, x0):
        z = self.solve_ode(t_ref, z, parameters, x0)
        N = np.array(z['y'][:-1, :])
        loss = N/np.sum(N, axis=0) - N_ref/np.sum(N, axis=0)
        return loss

    def lmfit_loss(self, params, t_ref, N_ref, x0, z):
        parameters = [params['ks'],
                      params['kp'],
                      params['kg'],
                      params['s'],
                      params['p'],
                      params['g'],
                      params['gamma'],
                      params['kv']]
        loss = self.loss(t_ref, N_ref, z, parameters, x0)
        return loss

    def plot_ode_solution(self, t, x):
        # Define process variables
        L = self.domain.axis[0].midpoints()
        N = x[:self.domain.axis[0].m,:]
        C = x[-2,:]
        T = x[-1,:]
        t = t
        # Calculate solubility curve
        C_sat = self.solute.solubility(T=T)

        # Create time and size mesh-grid
        t_mesh, L_mesh = np.meshgrid(t, L)

        # Plot 3d size-distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(t_mesh, L_mesh, N)
        ax.set(xlabel='Time [s]', ylabel=self.domain.axis[0].disc_by+' [µm]', zlabel='Particle density [-/µL]')

        # Plot concentration/temperature plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(T, C, label="Operating line")
        ax.plot(T, C_sat, label="Solubility curve")
        ax.legend()
        ax.set(xlabel='Temperature [C]', ylabel='Concentration [g/µL]')

        # Plot time - concentration,temperature plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlabel='Time [s]')
        color = 'tab:red'
        ax.plot(t, T, color=color)
        ax.set_ylabel('Temperature [C]', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax_right = ax.twinx()
        color = 'tab:blue'
        ax_right.plot(t, C, color=color)
        ax_right.set_ylabel('Concentration [g/µL]', color=color)
        ax_right.tick_params(axis='y', labelcolor=color)
