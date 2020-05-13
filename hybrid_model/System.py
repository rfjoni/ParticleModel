# -*- coding: utf-8 -*-
"""System module

This module contains the System and Sensor class for defining group of measured and controlled sensor variables

"""

# Importing dependencies
from typing import List
from domain.Domain import Domain


class LossSettings:
    """LossSettings class for containing loss settings

    Attributes:
        geometry: Geometry type
        loss_type: Loss basis
    """

    def __init__(self, geometry: str = 'Sphere', loss_type: str = 'Number') -> None:
        """
        Create LossSettings object

        Args:
            geometry: Geometry type
            loss_type: Loss basis
        """
        self.geometry = geometry
        self.loss_type = loss_type


class OdeSettings:
    """OdeSettings class for containing ode settings

    Attributes:
        variable_stepsize: Use variable step-size ode solver
        time_steps: Time-steps in ode solver
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
    """

    def __init__(self, variable_stepsize: bool = True, time_steps: int = 2000, rel_tol: float = 0.001,
                 abs_tol: float = 0.1) -> None:
        """
        Create OdeSettings object

        Args:
            variable_stepsize: Use variable step-size ode solver
            time_steps: Maximum time-steps in ode solver
            rel_tol: Relative tolerance
            abs_tol: Absolute tolerance
        """
        self.variable_stepsize = variable_stepsize
        self.time_steps = time_steps
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol


class RateSettings:
    """RateSettings class for containing rate model settings

    Attributes:
        scaling_factors: Specify scaling factors for phenomena rates
        layer_activations: Specify neural network layer activations
        layer_neurons: Specify neural network layer neurons
    """

    def __init__(self, layer_activations: list, layer_neurons: list, scaling_factors: dict = None) -> None:
        """
        Create RateSettings object

        Args:
            scaling_factors: Specify scaling factors for phenomena rates
            layer_activations: Specify neural network layer activations
            layer_neurons: Specify neural network layer neurons
        """
        self.scaling_factors = scaling_factors or {'nucleation': 1/100,
                                                   'growth': 1/10000,
                                                   'shrinkage': 1/10000,
                                                   'agglomeration': 1/100,
                                                   'breakage': 1/100}
        self.layer_activations = layer_activations
        self.layer_neurons = layer_neurons

    def set_scaling_factor(self, scaling_phenomena: str, scaling_factor: float) -> None:
        """
        Setting rate scaling factor

        Args:
            scaling_phenomena: Phenomena to set scaling factor
            scaling_factor: Scaling factor for phenomena
        """
        for item in scaling_phenomena:
            if item in self.scaling_factors:
                self.scaling_factors[item] = scaling_factor
                print('Scaling factor for ' + item + ' has been set to ' + str(scaling_factor))
            else:
                print(item, 'is not a valid phenomenon')

class Sensor:
    """Sensor class for containing sensor information

    Attributes:
        name: Sensor name (as stated in data)
        measured: Measured process variable
        controlled: Controllable process variable
        unit: Sensor input unit
    """

    def __init__(self, name: str, measured: bool, controlled: bool, unit: str) -> None:
        """
        Create sensor object

        Args:
            name: Sensor name (as stated in data)
            measured: Measured process variable
            controlled: Controllable process variable
            unit: Sensor input unit
        """
        self.name = name
        self.measured = measured
        self.controlled = controlled
        self.unit = unit


class System:
    """System class for containing model information

    Attributes:
        case: Case name
        phenomena: Phenomena
        sensors: Sensor list
        domain: Domain
        dilution: Apply dilution calculation
        regularization: Regularization constant
        normalize: Apply size distribution normalization
        ode_settings: Ode settings
        loss_settings: Loss settings
    """

    def __init__(self, case: str, domain: Domain, ode_settings: OdeSettings, loss_settings: LossSettings,
                 rate_settings: RateSettings, dilution: bool = True, regularization: float = 1,
                 normalize: bool = False) -> None:
        """
        Creating Sensor group

        Args:
            case: Case name
            domain: Domain object
            ode_settings: ODE settings object
            loss_settings: loss settings object
            dilution: Apply dilution calculation
            regularization: Regularization constant
            normalize: Apply size distribution normalization
        """
        self.case = case
        self.phenomena = {'nucleation': False,
                          'growth': False,
                          'shrinkage': False,
                          'agglomeration': False,
                          'breakage': False}
        self.sensors: List[Sensor] = []
        self.domain = domain
        self.dilution = dilution
        self.regularization = regularization
        self.normalize = normalize
        self.ode_settings = ode_settings
        self.loss_settings = loss_settings
        self.rate_settings = rate_settings
        print(self.case, 'has been successfully created')

    def add_sensor(self, name: str, measured: bool = True, controlled: bool = False, unit: str = '-') -> None:
        """
        Adding sensor to sensor group

        Args:
            name: Sensor name (as stated in data)
            measured: Measured process variable
            controlled: Controllable process variable
            unit: Sensor input unit
        """
        self.sensors.append(Sensor(name, measured, controlled, unit))
        print('Sensor', name, 'has been successfully added to', self.case)

    def activate_phenomena(self, phenomena: List[str]) -> None:
        """
        Activating phenomena in model

        Args:
            phenomena: List of phenomenon/phenomena to activate
        """
        for item in phenomena:
            if item in self.phenomena:
                self.phenomena[item] = True
                print(item, 'has been activated')
            else:
                print(item, 'is not a valid phenomenon')
