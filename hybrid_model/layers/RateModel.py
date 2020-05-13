# -*- coding: utf-8 -*-
"""Rate model layer

This module contains code for kinetic rate model

"""
# Importing dependencies
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Concatenate, Dense, Lambda, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from hybrid_model.System import System, RateSettings
from typing import Tuple


class RateModel():
    def __init__(self, system: System):
        self.system = system

    def create_rate_model(self, input_dimension: Tuple[int, int], rate_sizes: dict,
                          rate_settings: RateSettings, name: str) -> Model:
        # Get input vectors
        rate_model_input = Input(shape=(2, input_dimension[0],), name='Rate_input')
        distribution_input = Input(shape=(input_dimension[1],), name='Dist_input')
        if self.system.normalize:
            dist = Lambda(lambda x: tf.einsum('ki,k->ki', x, tf.math.reciprocal(tf.maximum(tf.ones_like(tf.reduce_sum(x, axis=[-1]), dtype=tf.float32), tf.reduce_sum(x, axis=[-1])))))(distribution_input)
        else:
            dist = distribution_input
        # Black box rate model
        flatten_input = Reshape(target_shape=(2*input_dimension[0],))(rate_model_input)
        total_input = Concatenate(axis=-1)([flatten_input, dist])
        batch_norm_input = BatchNormalization(name='Batch_norm_input')(total_input)
        hidden_layer = batch_norm_input
        for layer_number, (hidden_neurons, activation) in enumerate(zip(rate_settings.layer_neurons,
                                                                        rate_settings.layer_activations)):
            hidden_layer = Dense(hidden_neurons, name='Hidden_layer_'+str(layer_number),
                                 activation=activation)(hidden_layer)
        rates = Dense(sum(rate_sizes.values()), name='Phenomena_rate_layer')(hidden_layer)
        # Split rates depending on which phenomena that were selected
        rates = Lambda(lambda rates: tf.split(rates, [rate_sizes['nucleation'],
                                                      rate_sizes['growth'],
                                                      rate_sizes['shrinkage'],
                                                      rate_sizes['agglomeration'],
                                                      rate_sizes['breakage']], axis=1),
                       name='Rate_splitter')(rates)
        # Calculate individual rates
        nucleation_rate = Lambda(lambda rate: tf.abs(tf.multiply(rate,self.system.rate_settings.scaling_factors['nucleation'])),
                                 name='Nucleation_rate')(rates[0])
        growth_rate = Lambda(lambda rate: tf.abs(tf.multiply(rate, self.system.rate_settings.scaling_factors['growth'])),
                             name='Growth_rate')(rates[1])
        shrinkage_rate = Lambda(lambda rate: tf.abs(tf.multiply(rate, self.system.rate_settings.scaling_factors['shrinkage'])),
                             name='Shrinkage_rate')(rates[2])
        agglomeration_rate = Lambda(lambda rate: tf.abs(tf.multiply(rate, self.system.rate_settings.scaling_factors['agglomeration'])),
                             name='Agglomeration_rate')(rates[3])
        breakage_rate = Lambda(lambda rate: tf.abs(tf.multiply(rate, self.system.rate_settings.scaling_factors['breakage'])),
                               name='Breakage_rate')(rates[4])
        # Generate rate model
        rate_model = Model(inputs=[rate_model_input, distribution_input],
                           outputs=[nucleation_rate, growth_rate, shrinkage_rate, agglomeration_rate, breakage_rate],
                           name=name)
        #rate_model.summary()
        return rate_model