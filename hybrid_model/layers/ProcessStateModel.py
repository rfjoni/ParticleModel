# -*- coding: utf-8 -*-
"""Process state layer

This module contains a model for deriving process state variables and dz/dt

"""
# Importing dependencies
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from hybrid_model.System import System
from typing import List


class ProcessStateModel:
    def __init__(self, system: System):
        self.system = system

    def process_variable_state(self, tensor_input: List[tf.Tensor]) -> tf.Tensor:
        # Initialize list of controlled variables
        controlled_index = []
        measured_index = []
        # Get index of controlled variables in measured variable input
        for index, sensor in enumerate(self.system.sensors):
            if sensor.measured and sensor.controlled:
                controlled_index.append(index)
            if sensor.measured:
                measured_index.append(index)
        # Convert lists to tensor flow lists
        controlled_tf_index = tf.constant([[index] for index in controlled_index], dtype=tf.int32)  # [cv]

        # Define input vectors
        current_measured = tensor_input[0]  # shape (?,measured_variable_dimension)
        future_controlled = tensor_input[1]  # shape (?,controlled_variable_dimension)
        delta_time = tensor_input[2]  # shape (?,1)

        # Derivative of controlled variables
        current_controlled = tf.gather_nd(tf.transpose(current_measured), controlled_tf_index)
        future_controlled = tf.transpose(future_controlled)
        controlled_derivative = tf.divide(future_controlled - current_controlled, tf.transpose(delta_time))

        # Insert overall time derivative
        if len(controlled_index) > 0:
            process_variable_derivative = tf.scatter_nd(controlled_tf_index, controlled_derivative,
                                                        tf.shape(tf.transpose(current_measured)))
            process_variable_derivative = tf.transpose(process_variable_derivative)
        else:
            process_variable_derivative = tf.zeros_like(current_measured)
        return tf.stack([current_measured, process_variable_derivative], axis=1)

    def create_data_prep_model(self, measured_variable_dimension: int,
                               controlled_variable_dimension: int, name: str) -> Model:

        # Get input vectors
        measured_variable = Input(shape=(measured_variable_dimension,), name='Measured_variable')
        controlled_variable = Input(shape=(controlled_variable_dimension,), name='Controlled_variable')
        time = Input(shape=(1,), name='Time')

        # Get process variable values (current and derivative)
        process_variable_derivatives = Lambda(self.process_variable_state,
                                              output_shape=(2, measured_variable_dimension,),
                                              name='Controlled_variable_derivative')([measured_variable,
                                                                                      controlled_variable,
                                                                                      time])
        # Generate data preparation model
        data_prep_model = Model(inputs=[measured_variable, controlled_variable, time],
                                outputs=process_variable_derivatives, name=name)
        data_prep_model.units = {'input': '-', 'output': '-/mL'}
        return data_prep_model
