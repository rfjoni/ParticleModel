# -*- coding: utf-8 -*-
"""Hybrid model module

This module contains the model structure generator for the hybrid particle model

"""
# Importing dependencies

import logging
from hybrid_model.layers.PopulationBalanceModel import PopulationBalanceModel
from hybrid_model.layers.ConcentrationModel import ConcentrationModel
from hybrid_model.layers.UncertaintyModel import UncertaintyModel
from hybrid_model.layers.RateModel import RateModel
from hybrid_model.layers.ProcessStateModel import ProcessStateModel
from hybrid_model.loss.HybridModelLoss import HybridModelLoss
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape
from tensorflow.keras.models import Model
from hybrid_model.System import System
from typing import Dict, List, Tuple
import tensorflow as tf
import numpy as np
import pickle
import gc
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class HybridModel:
    def __init__(self, system: System) -> None:
        # Create sub-models dictionary and model
        self.sub_models = {}
        self.training_model: Model
        self.prediction_model: Model
        self.multi_step_model: Model
        self.system = system
        # Get input size based on definitions in system object
        self.input_size = self.input_size(system)
        # Get phenomena rate sizes
        self.rate_sizes = self.rate_sizes(system)
        # Initialize loss model
        self.loss_model = None
        # Set up hybrid sub-models
        self.create_sub_models()
        # Set up overall hybrid model
        self.create_hybrid_model()
        self.ref_loss_train = None
        self.ref_loss_val = None
        self.training_history = None

    @staticmethod
    def input_size(system: System) -> Dict[str, int]:
        measured_variable_dimension = 0
        controlled_variable_dimension = 0
        distribution_dimension = system.domain.axis[0].m
        # Measured and controlled variables
        for sensor in system.sensors:
            # If sensor is selected as measured
            if sensor.measured:
                measured_variable_dimension += 1
            # If sensor is selected as controlled
            if sensor.controlled:
                controlled_variable_dimension += 1
        return {'measured_variable': measured_variable_dimension,
                'controlled_variable': controlled_variable_dimension,
                'distribution': distribution_dimension}

    @staticmethod
    def rate_sizes(system: System) -> Dict[str, int]:
        rate_sizes = {'nucleation': 1 * system.phenomena['nucleation'],
                      'growth': system.domain.axis[0].m * system.phenomena['growth'],
                      'shrinkage': system.domain.axis[0].m * system.phenomena['shrinkage'],
                      'agglomeration': round(system.domain.axis[0].m * (system.domain.axis[0].m + 1) / 2) *
                                       system.phenomena['agglomeration'],
                      'breakage': round(
                          system.domain.axis[0].m * (system.domain.axis[0].m - 1) / 2 + system.domain.axis[0].m) *
                                  system.phenomena['breakage']}
        return rate_sizes

    def create_sub_models(self):
        # Generate uncertainty models
        self.sub_models['Distribution uncertainty'] = \
            UncertaintyModel(system=self.system).create_uncertainty_model(variable_dimension=self.input_size['distribution'],
                                                                          uncertainty_type='distribution',
                                                                          name='Uncertainty_dist')
        self.sub_models['Measured variable uncertainty'] = \
            UncertaintyModel(system=self.system).create_uncertainty_model(variable_dimension=self.input_size['measured_variable'],
                                                                          uncertainty_type='ordinal',
                                                                          name='Uncertainty_PV')
        self.sub_models['Controlled variable uncertainty'] = \
            UncertaintyModel(system=self.system).create_uncertainty_model(variable_dimension=self.input_size['controlled_variable'],
                                                                          uncertainty_type='ordinal',
                                                                          name='Uncertainty_CV')
        # Generate dilution models
        self.sub_models['Dilution in'] = \
            ConcentrationModel().create_dilution_model(variable_dimension=self.input_size['distribution'],
                                                       correction_type='in',
                                                       name='Concentration_now')
        self.sub_models['Dilution out'] = \
            ConcentrationModel().create_dilution_model(variable_dimension=self.input_size['distribution'],
                                                       correction_type='out',
                                                       name='Concentration_future')
        # Generate process state model
        self.sub_models['Process states'] = \
            ProcessStateModel(system=self.system).create_data_prep_model(measured_variable_dimension=self.input_size['measured_variable'],
                                                                         controlled_variable_dimension=self.input_size['controlled_variable'],
                                                                         name='Process_states')
        # Generate rate model
        self.sub_models['Rate'] = \
            RateModel(system=self.system).create_rate_model(input_dimension=(self.input_size['measured_variable'],
                                                                             self.input_size['distribution']),
                                                            rate_sizes=self.rate_sizes,
                                                            rate_settings = self.system.rate_settings,
                                                            name='Rate_model')

        # Generate PBM model
        self.sub_models['Population Balance Model'] = \
            PopulationBalanceModel(system=self.system,
                                   rate_model=None).create_pbm_model(distribution_dimension=self.input_size['distribution'],
                                                                     process_variable_dimension=self.input_size['measured_variable'],
                                                                     rate_sizes=self.rate_sizes,
                                                                     name='Population_balance_model_train')
        self.sub_models['Population Balance Model Prediction'] = \
            PopulationBalanceModel(system=self.system,
                                   rate_model=self.sub_models['Rate']).create_pbm_model(distribution_dimension=self.input_size['distribution'],
                                                                                        process_variable_dimension=self.input_size['measured_variable'],
                                                                                        rate_sizes=self.rate_sizes,
                                                                                        name='Population_balance_model_pred')
        # Generate loss object
        self.loss_model = HybridModelLoss(system=self.system)

    def create_hybrid_model(self):
        # Define hybrid model inputs and input size
        distribution_now = Input(shape=(self.input_size['distribution'],), name='Distribution_now')
        measured_variable = Input(shape=(self.input_size['measured_variable'],), name='Measured_variable')
        controlled_variable = Input(shape=(self.input_size['controlled_variable'],), name='Controlled_variable')
        measured_variable_unc = Input(shape=(self.input_size['measured_variable'],), name='Measured_variable_unc')
        controlled_variable_unc = Input(shape=(self.input_size['controlled_variable'],), name='Controlled_variable_unc')
        dilution_factor_now = Input(shape=(2,), name='Dilution_factor_now')
        dilution_factor_future = Input(shape=(2,), name='Dilution_factor_future')
        time = Input(shape=(1,), name='Time')

        # Model structure definition
        uncertain_distribution = self.sub_models['Distribution uncertainty']([distribution_now])
        uncertain_measured_variable = self.sub_models['Measured variable uncertainty']([measured_variable, measured_variable_unc])
        uncertain_controlled_variable = self.sub_models['Controlled variable uncertainty']([controlled_variable, controlled_variable_unc])
        distribution_concentration = self.sub_models['Dilution in']([uncertain_distribution, dilution_factor_now])
        process_variables = self.sub_models['Process states']([uncertain_measured_variable, uncertain_controlled_variable, time])
        rates = self.sub_models['Rate']([process_variables, distribution_concentration])
        pbm_prediction = self.sub_models['Population Balance Model']([distribution_concentration, process_variables, time, rates])[0]
        predicted_number_distribution = self.sub_models['Dilution out']([pbm_prediction, dilution_factor_future])
        # Define training model
        self.training_model = Model(inputs=[distribution_now,
                                            measured_variable,
                                            controlled_variable,
                                            measured_variable_unc,
                                            controlled_variable_unc,
                                            dilution_factor_now,
                                            dilution_factor_future,
                                            time],
                                    outputs=predicted_number_distribution)
        self.training_model.summary()

    def create_multi_step_model(self):
        # Extract rate model without training abilities
        #self.sub_models['Rate'].trainable = False

        # Multi step inputs
        distribution = Input(shape=(self.input_size['distribution'],), name='Distribution')
        process_variables = Input(shape=(self.input_size['measured_variable'],), name='Process_variables')
        controlled_derivative = Input(shape=(self.input_size['controlled_variable'],), name='Controlled_derivative') # [?]
        time_step = Input(shape=(1,), name='Time_step')

        # Define model
        cv_future = Lambda(lambda x: tf.transpose(tf.expand_dims(x[0][:, 0], axis=0) + tf.transpose(x[1][:, :])*x[2][:]),
                           output_shape=(self.input_size['controlled_variable'],))([process_variables,
                                                                                    controlled_derivative,
                                                                                    time_step])
        process_states = self.sub_models['Process states']([process_variables, cv_future, time_step])
        rates = self.sub_models['Rate']([process_states, distribution])
        pbm_prediction = self.sub_models['Population Balance Model Prediction']([distribution,
                                                                                 process_states,
                                                                                 time_step,
                                                                                 rates])
        self.multi_step_model = Model(inputs=[distribution, process_variables, controlled_derivative, time_step],
                                      outputs=pbm_prediction, name='Multi_step_model')
        self.multi_step_model.summary()


    def multi_step_prediction(self, steps):
        # Initial conditions
        initial_distribution = Input(shape=(self.input_size['distribution'],), name='Initial_distribution')
        initial_dilution_factor = Input(shape=(2,), name='Initial_dilution_factor')
        initial_process_variables = Input(shape=(self.input_size['measured_variable'],), name='Initial_process_variables')
        initial_psd_distribution = self.sub_models['Dilution in']([initial_distribution, initial_dilution_factor])
        # Target conditions
        target_distribution = Input(shape=(self.input_size['distribution'],), name='Target_distribution')
        target_dilution_factor = Input(shape=(2,), name='Target_dilution_factor')
        #target_process_variables = Input(shape=(self.input_size['measured_variable'],), name='Target_process_variables')
        target_psd_distribution = self.sub_models['Dilution in']([target_distribution, target_dilution_factor])
        # Dynamic conditions
        controlled_variables = Input(shape=(steps, self.input_size['controlled_variable'],), name='Controlled_variable')
        time_steps = Input(shape=(steps), name='Time_steps')
        # Set initial conditions
        process_variables = [initial_process_variables]
        psd_distribution = [initial_psd_distribution]
        rates = []
        # Multi step prediction
        for step in range(steps):
            # Current CV
            current_controlled_variables = Lambda(lambda x: x[:, step, :])(controlled_variables)
            current_time_step = Lambda(lambda x: x[:, step])(time_steps)
            process_states = self.sub_models['Process states']([process_variables[-1], current_controlled_variables, current_time_step])
            rates.append(self.sub_models['Rate']([process_states, psd_distribution[-1]]))
            pbm_prediction = self.sub_models['Population Balance Model Prediction']([psd_distribution[-1],
                                                                                     process_states,
                                                                                     current_time_step,
                                                                                     rates[-1]])
            psd_distribution.append(pbm_prediction[0])
            process_variables.append(pbm_prediction[1])

        multi_step_prediction = Model(inputs=[initial_distribution,
                                              initial_dilution_factor,
                                              initial_process_variables,
                                              target_distribution,
                                              target_dilution_factor,
                                              #target_process_variables,
                                              controlled_variables,
                                              time_steps],
                                      outputs=[psd_distribution[-1],
                                               target_psd_distribution,
                                               initial_psd_distribution,
                                               process_variables[-1],
                                               rates,
                                               ],
                                      name='Multi_step_prediction')
        multi_step_prediction.summary()
        return multi_step_prediction

    def calculate_reference_loss(self, x, y, category: str):
        y_tensor = tf.constant(y, dtype=tf.float32)
        x_tensor = tf.constant(np.transpose(np.transpose(x[0]) / (x[5][:, 0] * x[5][:, 1]) * x[6][:, 0] * x[6][:, 1]),
                               dtype=tf.float32)
        reference_loss = self.loss_model.loss(y_tensor, x_tensor).numpy()
        av_reference_loss = np.mean(reference_loss)
        if category == 'train':
            self.ref_loss_train = {'Reference loss': reference_loss,
                                   'Average loss': av_reference_loss}
            print('Training reference error: ' + str(self.ref_loss_train['Average loss']))
        elif category == 'val':
            self.ref_loss_val = {'Reference loss': reference_loss,
                                 'Average loss': av_reference_loss}
            print('Validation reference error: ' + str(self.ref_loss_val['Average loss']))
        else:
            print('category not properly set')

    @staticmethod
    def model_data(shuffled_data):
        x = [shuffled_data['Current distribution'],
             shuffled_data['Measured variables'],
             shuffled_data['Controlled variables'],
             shuffled_data['Measured variable uncertainty'],
             shuffled_data['Controlled variable uncertainty'],
             shuffled_data['Current dilution factor'],
             shuffled_data['Future dilution factor'],
             shuffled_data['Time horizon']]
        y = shuffled_data['Future distribution']
        return [x, y]

    def save_training_history(self, filename='training_history'):
        with open(filename+'.pkl', 'wb') as saved:
            pickle.dump([self.ref_loss_train, self.ref_loss_val, self.training_history],
                        saved, pickle.HIGHEST_PROTOCOL)

    def load_training_history(self, filename='training_history'):
        with open(filename + '.pkl', 'rb') as loaded:
            self.ref_loss_train, self.ref_loss_val, self.training_history = pickle.load(loaded)