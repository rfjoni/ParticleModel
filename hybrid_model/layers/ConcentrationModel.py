# -*- coding: utf-8 -*-
"""Concentration layer

This module contains code for calculating PSD concentration based on dilution and volume

"""
# Importing dependencies
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model


class ConcentrationModel:
    """
    """
    @staticmethod
    def correction_in(concentration_input):
        # Split input
        y = concentration_input[0]
        correction_factor = concentration_input[1][:, 0]
        flowcell_volume = concentration_input[1][:, 1]
        # Return conversion factor
        return y/tf.reshape(correction_factor*flowcell_volume, (-1, 1))

    @staticmethod
    def correction_out(concentration_input):
        # Split input
        y = concentration_input[0]
        correction_factor = concentration_input[1][:, 0]
        flowcell_volume = concentration_input[1][:, 1]
        # Return conversion factor
        return y*tf.reshape(correction_factor*flowcell_volume, (-1, 1))

    def create_dilution_model(self, variable_dimension: int, correction_type: str, name: str) -> Model:
        # Get input vector
        variable = Input(shape=(variable_dimension,), name='Variable')
        dilution_factor = Input(shape=(2,), name='Dilution_factor')
        # Create concentration correction layer
        if correction_type == 'in':
            corrected_variable = Lambda(self.correction_in,
                                        output_shape=self.output_shape,
                                        name='Corrected_variable')([variable, dilution_factor])
        elif correction_type == 'out':
            corrected_variable = Lambda(self.correction_out,
                                        output_shape=self.output_shape,
                                        name='Corrected_variable')([variable, dilution_factor])
        else:
            raise ValueError("Dilution model correction type not properly set")
        # Generate dilution model
        dilution_model = Model(inputs=[variable, dilution_factor], outputs=[corrected_variable], name=name)
        dilution_model.units = {'input': '-', 'output': '-/mL'}
        return dilution_model

    @staticmethod
    def output_shape(input_shape):
        return tuple(input_shape)
