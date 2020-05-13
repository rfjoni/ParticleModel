# -*- coding: utf-8 -*-
"""Uncertainty layer

This module contains a keras model for estimating and adding sampling error to a PSD

"""

# Importing dependencies
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from hybrid_model.System import System
from hybrid_model.layers.NoiseModel import NoiseModel


class UncertaintyModel:
    """
    Uncertainty object for estimating measurement uncertainty
    """
    def __init__(self, system: System):
        self.system = system

    @staticmethod
    def sampling_error(y: tf.Tensor) -> tf.Tensor:
        """
        Calculates the standard sampling error

        Args:
            y: Number based size distribution
        Returns:
            tensor: Standard error for each bin of the size distribution
        """

        # Find number of particles per batch size
        number_particles = tf.reduce_sum(y, axis=[-1])
        # In case of no particles, set to unity to avoid zero-division
        number_particles = tf.maximum(tf.ones_like(number_particles, dtype=tf.float32), number_particles)
        # Calculate standard error
        error = tf.sqrt(y * (1 - y / tf.reshape(number_particles, (-1, 1))))
        return error

    @staticmethod
    def output_shape(input_shape):
        return input_shape

    def create_uncertainty_model(self, variable_dimension: int,
                                 uncertainty_type: str, name: str) -> Model:
        # Get input vector
        variable = Input(shape=(variable_dimension,), name='Variable')
        # Set or calculate uncertainty
        if uncertainty_type == 'distribution':
            uncertainty = Lambda(self.sampling_error,
                                 output_shape=[(None, variable_dimension)],
                                 name='Variable_uncertainty')(variable)
        elif uncertainty_type == 'ordinal':
            uncertainty = Input(shape=(variable_dimension,), name='Variable_uncertainty')
        else:
            raise ValueError("Uncertainty model type not properly set")
        # Create noise object
        noise = NoiseModel(noise_factor=self.system.regularization)
        # Add uncertainty as noise
        uncertain_variable = Lambda(noise.noisy_signal,
                                    name='Uncertain_variable')([variable,
                                                               uncertainty])
        if uncertainty_type == 'distribution':
            uncertainty_model = Model(inputs=[variable], outputs=[uncertain_variable], name=name)
        elif uncertainty_type == 'ordinal':
            uncertainty_model = Model(inputs=[variable, uncertainty], outputs=[uncertain_variable], name=name)
        else:
            raise ValueError("Uncertainty model type not properly set")
        #uncertainty_model.units = {'input': '-', 'output': '-/mL'}
        #uncertainty_model.summary()
        return uncertainty_model