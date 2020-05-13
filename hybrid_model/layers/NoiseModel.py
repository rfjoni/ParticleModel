# -*- coding: utf-8 -*-
"""Noise layer

This module contains code for adding uncertainty based noise to measured process variables

"""

# Importing dependencies
import tensorflow as tf
from typing import List
import tensorflow_probability as tp


class NoiseModel:
    """
    Object for generating noise based on measurement uncertainty
    """
    def __init__(self, noise_factor=1):
        self.noise_factor = noise_factor

    def batch_noise(self, std_error: tf.Tensor) -> tf.Tensor:
        """
        Generates random batch noise based on sampling error a batch of size distributions

        Args:
            std_error (object): Standard error of a single train entry
        Returns:
            object: Batch of random noise based on sampling error of each bin of the size distribution
        """
        # Get batch size
        batch_size = tf.shape(std_error)[0]
        # Define train-sets to loop over
        train_indexes = tf.range(batch_size)
        # Define loop function
        fun = lambda train_index: self.train_noise(train_index, std_error)
        # Run loop
        batch_noise = tf.map_fn(fun, train_indexes, dtype=tf.float32)
        return batch_noise

    def train_noise(self, train_index: tf.Tensor, std_error: tf.Tensor) -> tf.Tensor:
        """
        Calculates random noise for a single training entry

        Args:
            train_index: Index of training entry
            std_error: Standard error of batch data
        Returns:
            object: Single train entry random noise based on sampling error of each bin of the size distribution
        """

        # Get number of features
        features = tf.shape(std_error)[1]
        # Define bins to loop over
        feature_indexes = tf.range(features)
        # Define loop function
        fun = lambda feature_index: self.feature_noise(feature_index, train_index, std_error)
        # Run loop
        train_noise = tf.map_fn(fun, feature_indexes, dtype=tf.float32)
        return train_noise

    def feature_noise(self, feature_index: tf.Tensor, train_index: tf.Tensor, std_error: tf.Tensor) -> tf.Tensor:
        """
        Calculates random noise for a single training entry for a single bin

        Args:
            feature_index: Index of feature
            train_index: Index of train entry
            std_error: Standard error of a single train entry
        Returns:
            tensor: Single train entry random noise based on standard error
        """

        # Generate random noise
        noise = tp.distributions.Normal(loc=0, scale=std_error[train_index, feature_index]).sample(sample_shape=[])
        return noise

    def noisy_signal(self, noise_input: List[tf.Tensor]) -> tf.Tensor:
        # Split input
        y = noise_input[0]
        std_error = noise_input[1]
        # Obtain batch noise
        batch_noise = self.batch_noise(std_error)
        # Add noise to signal
        noisy_distribution = tf.add(y, tf.multiply(batch_noise, self.noise_factor))
        # Ensure no PSD lower than 0
        noisy_distribution = tf.maximum(tf.zeros_like(y), noisy_distribution)
        return tf.keras.backend.in_train_phase(noisy_distribution, y)

    @staticmethod
    def output_shape(input_shape):
        return input_shape[0]
