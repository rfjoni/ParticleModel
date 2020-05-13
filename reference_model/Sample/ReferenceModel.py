# -*- coding: utf-8 -*-
"""Reference model module

This module contains code for reference model

"""
# Importing dependencies
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime
from data.Data import Measurement, Sensor
from reference_model.PseudoSensors import PseudoSensors


class ReferenceModel:
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
                elif kinetic_name == 'Shrinkage':
                    # Set birth and death rate
                    gd = N*rate/(2*self.domain.axis[0].widths())
                    birth_rate = np.concatenate([gd[1:], [0]])
                    death_rate = gd
                elif kinetic_name == 'Agglomeration':
                    pass
                elif kinetic_name == 'Breakage':
                    pass
                bin_rates[kinetic_name] = birth_rate - death_rate
        return bin_rates

    def dxdt(self, x, t, z):
        # Initialize dNdt and dTdt
        dNdt = np.zeros([self.domain.axis[0].m])
        dTdt = z[0]

        # Split x to N, C and T
        N = x[:self.domain.axis[0].m]
        C = x[-2]
        T = x[-1]

        # Calculate sigma and total volume
        sigma = self.solute.supersaturation(C=C, T=T)
        vc = np.sum(N*self.domain.axis[0].midpoints()**3)

        # Calculate phenomena kinetics
        kinetics = self.kinetics.rates(sigma=sigma, vc=vc, lengths=self.domain.axis[0].midpoints())

        # Rates
        bin_rates = self.rates(kinetics=kinetics, N=N)

        # Calculate dNdt
        for kinetic_name, bin_rate in bin_rates.items():
            dNdt = dNdt + bin_rate

        # Calculate dCdt
        dCdt = -self.solute.crystal_density*self.solute.crystal_shape_factor * \
            np.sum(dNdt*self.domain.axis[0].midpoints()**3)

        # Collect as dxdt
        dxdt = np.concatenate([dNdt, [dCdt], [dTdt]])

        return dxdt

    def solve_ode(self, t0, timestep, z, x0):
        # Solve ode
        t_eval = np.linspace(t0, t0+timestep, 20)
        z = solve_ivp(fun = lambda t, x: self.dxdt(x, t, z),
                      y0=x0, t_span=(t0, t0+timestep),
                      method='LSODA', t_eval=t_eval)
        return z

    def measurement_object(self, t, x, noise_level=1):
        # Extract process states at end time
        N = x[:self.domain.axis[0].m, -1]
        C = x[-2,-1]
        T = x[-1,-1]
        t = t[-1]
        # Create pseudo sensor object
        pseudo_sensors = PseudoSensors(domain=self.domain, noise_level=noise_level)
        # Add noise to measurements
        N_noise = pseudo_sensors.PSD_noise(N)
        C_noise = pseudo_sensors.sensor_noise(C, stdev=0.01)
        T_noise = pseudo_sensors.sensor_noise(T, stdev=0.1)
        # Create pseudo image analysis sensor results
        ia_data = pseudo_sensors.image_analysis(N_noise)
        # Generate artificial time
        time_stamp = datetime.fromtimestamp(t+1567870637)
        # Create measurement object with data
        measurement = Measurement(measurement_id='', time=time_stamp)
        # External sensors
        temperature_sensor = Sensor(sensor_id = 'Temperature',
                                    value = T_noise,
                                    data_type='external',
                                    std_error=0.1*noise_level,
                                    unit = 'C')
        concentration_sensor = Sensor(sensor_id = 'Concentration',
                                      value= C_noise,
                                      data_type='external',
                                      std_error=0.01*noise_level,
                                      unit = 'g/µL')
        # Image analysis sensor
        image_analysis_sensor = Sensor(sensor_id=self.domain.axis[0].disc_by,
                                       value=ia_data,
                                       data_type='image_analysis',
                                       unit='µm')
        # Add sensor readings to measurement object
        measurement.add_external_sensor(temperature_sensor)
        measurement.add_external_sensor(concentration_sensor)
        measurement.add_particle_analysis_sensor(image_analysis_sensor)
        return measurement

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
