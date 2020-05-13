# -*- coding: utf-8 -*-
"""Hybrid model loss module

This module contains a customized model loss for training of hybrid model

"""

# Importing dependencies
import tensorflow as tf
import numpy as np
from hybrid_model.System import System
from hybrid_model.layers.UncertaintyModel import UncertaintyModel
from hybrid_model.layers.NoiseModel import NoiseModel


class HybridModelLoss:
    """HybridModelLoss class

    Attributes:
        system: Model system settings
    """

    def __init__(self, system: System):
        """
        Creating loss object

        Args:
            system (object): System specifications
        """
        self.system = system

    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate model loss

        Args:
            y_true: Experimentally measured size distribution
            y_pred: Predicted size distribution
        Returns:
            tensor: Tensor object with mean absolute error loss
        """

        # If adding noise to target is enabled
        if self.system.regularization > 0:
            uncertainty = UncertaintyModel.sampling_error(y_true)
            y_true = NoiseModel(noise_factor=self.system.regularization).noisy_signal([y_true, uncertainty])

        # If normalizing option is enabled
        if self.system.normalize:
            y_true = self.normalizer(y_true)
            y_pred = self.normalizer(y_pred)

        # Calculate absolute mean error loss
        loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[-1])

        if self.system.normalize:
            loss = loss / 2 * 100
        else:
            number_particles = tf.reduce_sum(y_true, axis=[-1])
            number_particles = tf.maximum(tf.ones_like(number_particles, dtype=tf.float32), number_particles)
            loss = loss / number_particles
        return loss

    def convert_distribution(self, y: tf.Tensor, probability_type: str, geometry: str) -> tf.Tensor:
        """
        Converts size distribution to number-, area-, or volume basis

        Args:
            y (object): Number based size distribution
            probability_type (str): Target type of size distribution (Number, Area, Volume)
            geometry (str): Assumed particle geometry (Sphere, Cube, Rod, Rod)
        Returns:
            object: Converted size distribution in desired target type
        """
        # Get number of axis in domain
        number_axis = self.system.domain.axis_counter

        if probability_type == 'Number':
            pass
        elif probability_type == 'Area':
            if number_axis == 1:
                if geometry == 'Sphere':
                    area = tf.convert_to_tensor(4 * np.pi * (self.system.domain.axis[0].midpoints() / 2) ** 2,
                                                dtype=tf.float32)
                elif geometry == 'Cube':
                    area = tf.convert_to_tensor(6 * self.system.domain.axis[0].midpoints() ** 2, dtype=tf.float32)
                else:
                    ValueError('Defined geometry invalid for number of axis')
                    area = y
                y = tf.einsum('ki,i->ki', y, area)
            elif number_axis == 2:
                if geometry == 'Rod':
                    a1 = 4 * np.outer(self.system.domain.axis[0].midpoints(), self.system.domain.axis[1].midpoints())
                    a2 = 2 * self.system.domain.axis[1].midpoints() ** 2
                    area = tf.convert_to_tensor(a1 + a2, dtype=tf.float32)
                else:
                    ValueError('Defined geometry invalid for number of axis')
                    area = y
                y = tf.einsum('kij,ij->kij', y, area)
        elif probability_type == 'Volume':
            if number_axis == 1:
                if geometry == 'Sphere':
                    volume = tf.convert_to_tensor(4 / 3 * np.pi * (self.system.domain.axis[0].midpoints() / 2) ** 3,
                                                  dtype=tf.float32)
                elif geometry == 'Cube':
                    volume = tf.convert_to_tensor(self.system.domain.axis[0].midpoints() ** 3, dtype=tf.float32)
                else:
                    ValueError('Defined geometry invalid for number of axis')
                    volume = y
                y = tf.einsum('ki,i->ki', y, volume)
            elif number_axis == 2:
                if geometry == 'Rod':
                    volume = tf.convert_to_tensor(
                        np.outer(self.system.domain.axis[0].midpoints(), self.system.domain.axis[1].midpoints() ** 2),
                        dtype=tf.float32)
                else:
                    ValueError('Defined geometry invalid for number of axis')
                    volume = y
                y = tf.einsum('kij,ij->kij', y, volume)
        return y

    def normalizer(self, y: tf.Tensor) -> tf.Tensor:
        """
        Normalizes the size-distribution y

        Args:
            y: Size distribution
        Returns:
            tensor: Normalized size distribution
        """

        # Get number of axis in domain
        number_axis = self.system.domain.axis_counter

        # Convert size distribution to desired basis
        y_converted = self.convert_distribution(y, probability_type=self.system.loss_settings.loss_type,
                                                geometry=self.system.loss_settings.geometry)

        # Normalize size distribution
        if number_axis == 1:
            number_particles = tf.reduce_sum(y_converted, axis=[-1])
            number_particles = tf.maximum(tf.ones_like(number_particles, dtype=tf.float32), number_particles)
            normalized_distribution = tf.einsum('ki,k->ki', y_converted, tf.math.reciprocal(number_particles))
        elif number_axis == 2:
            number_particles = tf.reduce_sum(tf.reduce_sum(y_converted, axis=[-1]), axis=[-1])
            number_particles = tf.maximum(tf.ones_like(number_particles, dtype=tf.float32), number_particles)
            normalized_distribution = tf.einsum('kij,k->kij', y_converted, tf.math.reciprocal(number_particles))
        else:
            ValueError('This code does not support above 2 dimensional particle shapes')
            normalized_distribution = y
        return normalized_distribution
