# -*- coding: utf-8 -*-
"""Population Balance Model layer

This module contains the PBM_loss and PBM_model classes, that are used as custom layers in tensorflow neural network

"""

# Importing dependencies
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from hybrid_model.System import System
from typing import Dict, List
from tensorflow.keras.layers import Input, Lambda
import tensorflow_scientific as ts
import tensorflow_probability as tfp


class PopulationBalanceModel:
    """Population balance model layer

    Attributes:
        system: System object
        rate_sizes: Dictionary of rate-sizes for each phenomena
        agglomeration_contribution: Agglomeration contribution constant matrix
        rate_model: External rate_model (not required for model training)

    """

    def __init__(self, system: System, rate_model: Model = None):
        """
        Creating tensorflow model object

        Args:
            system: System object with model specifications
            rate_sizes: Dictionary of rate-sizes for each phenomena
            rate_model: Keras model object (not required)
        """
        self.system = system
        self.rate_model = rate_model
        # Run agglomeration contribution calculation
        self.agglomeration_contribution = self.agglomeration_constant()


    def batch(self, tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Call PopulationBalanceModel ODE solver for batch data

        Args:
            List of tensors: List of tensor, [0]: rates (y), [1]: initial conditions (x0),
                            [2]: time horizon (delta t), [3]: dilution factor
        Returns:
            List of tensors: List of tensors [0]: PSD after time horizon, [1] additional PV's after time horizon
        """
        # Get batch size
        batch_size = tf.shape(tensors[0])[0]
        # Define loop function
        fun = lambda train_index: self.ode_solver(train_index, tensors)
        # Define batch elements to loop over
        train_indexes = tf.range(batch_size)
        # Run loop for batch size
        x_out = tf.map_fn(fun, train_indexes, dtype=tf.float32)
        # Extract PSD distribution (ensure solution contains no negative)
        n_out = tf.nn.relu(x_out[..., :self.system.domain.axis[0].m])
        # Get remaining process state variable predictions
        x_out_pv = x_out[..., self.system.domain.axis[0].m:]
        return [n_out, x_out_pv]

    def ode_solver(self, train_index: tf.Tensor, tensors: List[tf.Tensor]) -> tf.Tensor:
        """
        Run ODE solver for PopulationBalanceModel for single training example with index train_index

        Args:
            train_index: Training example index tensor
            tensors: List of tensors: List of tensor, [0]: rates (y), [1]: initial conditions (x0),
                            [2]: time horizon (delta t), [3]: dilution factor
        Returns:
            tensor: Tensor of process state predictions after time horizon
        """
        # Unpack and define the training example data
        n0 = tensors[0][train_index, :]  # Initial PSD
        process_variables_initial = tensors[1][train_index, 0, :]  # Initial PV's
        process_variables_derivative = tensors[1][train_index, 1, :]  # PV time-derivatives
        dt = tensors[2][train_index]  # Time horizon
        nucleation_rates = tensors[3][train_index, :]  # Rate tensor for nucleation rate
        growth_rates = tensors[4][train_index, :]  # Rate tensor for growth rate
        shrinkage_rates = tensors[5][train_index, :]  # Rate tensor for shrinkage rate
        agglomeration_rates = tensors[6][train_index, :]  # Rate tensor for agglomeration rate
        breakage_rates = tensors[7][train_index, :]  # Rate tensor for breakage rate
        # Initial time for ODE
        t0 = tf.zeros([], dtype=tf.float32)
        # Initial state (merge n0 and initial process variables)
        x0 = tf.concat([n0, process_variables_initial], axis=0)
        # Solve system of ODE equations
        x1 = tfp.math.ode.BDF().solve(lambda t, x: self.ode(x, t, process_variables_derivative,
                                                            nucleation_rates, growth_rates,
                                                            shrinkage_rates, agglomeration_rates,
                                                            breakage_rates),
                                      initial_time=0, initial_state=x0, solution_times=dt).states
        # Extract solution for t=t+dt
        x1 = tf.reshape(x1[-1, :], [tf.size(x0)])
        return x1

    def ode(self, x: tf.Tensor, t: tf.Tensor, process_variables_derivative: tf.Tensor,
            nucleation_rate: tf.Tensor, growth_rate: tf.Tensor, shrinkage_rate: tf.Tensor,
            agglomeration_rate: tf.Tensor, breakage_rate: tf.Tensor) -> tf.Tensor:
        """
        Calculate RHS of PBM differential equations

        Args:
            x (tensor): State tensor
            t (tensor): Time tensor
            process_variables_derivative (tensor): PV time-derivatives
            nucleation_rate (tensor): Rate tensor for nucleation rate
            growth_rate (tensor): Rate tensor for growth rate
            shrinkage_rate (tensor): Rate tensor for shrinkage rate
            agglomeration_rate (tensor): Rate tensor for agglomeration rate
            breakage_rate (tensor): Rate tensor for breakage rate
        Returns:
            tensor: dxdt tensor with time-derivative of all state variables x
        """

        # Initialize RHS
        n = x[:self.system.domain.axis[0].m]
        x_pv = x[self.system.domain.axis[0].m:]
        dndt = tf.zeros(self.system.domain.axis[0].m, dtype=tf.float32)
        dx_pvdt = process_variables_derivative

        # If prediction model prediction of rates
        # if self.rate_model is not None:
        #     rate_model_input = [tf.expand_dims(tf.stack([x_pv, dx_pvdt], axis=0), axis=0), tf.expand_dims(n, axis=0)]
        #     rates = self.rate_model(rate_model_input)
        #     nucleation_rate = rates[0][0]
        #     growth_rate = rates[1][0]
        #     shrinkage_rate = rates[2][0]
        #     agglomeration_rate = rates[3][0]
        #     breakage_rate = rates[4][0]

        # Phenomena rates
        if self.system.phenomena['nucleation']:
            dndt = dndt + self.nucleation(n, nucleation_rate)
        if self.system.phenomena['growth']:
            dndt = dndt + self.growth(n, growth_rate)
        if self.system.phenomena['shrinkage']:
            dndt = dndt + self.shrinkage(n, shrinkage_rate)
        if self.system.phenomena['agglomeration']:
            dndt = dndt + self.agglomeration(n, agglomeration_rate)
        if self.system.phenomena['breakage']:
            dndt = dndt + self.breakage(n, breakage_rate)
        #dx_pvdt = dx_pvdt + tf.constant([0.0, -1.52/(4500*1408*800)],
        #                                dtype=tf.float32)*tf.reduce_sum(dndt*tf.constant(self.system.domain.axis[0].midpoints()**3, dtype=tf.float32))
        return tf.concat([dndt, dx_pvdt], axis=-1)

    def nucleation(self, n, rate):
        # Nucleation calculations
        nucleation_factor = rate
        # Add contribution to birth and death
        birth = tf.concat([nucleation_factor, tf.zeros(self.system.domain.axis[0].m-1, dtype=tf.float32)], axis=0)
        return birth

    def growth(self, n, rate):
        # Growth calculations
        growth_factor = n / (2 * self.system.domain.axis[0].widths()) * rate
        # Add contribution to birth and death
        birth = tf.concat([tf.zeros(1, dtype=tf.float32), growth_factor[:-1]], axis=0)
        death = tf.concat([growth_factor[:-1], tf.zeros(1, dtype=tf.float32)], axis=0)
        return birth - death

    def shrinkage(self, n, rate):
        # Shrinkage calculations
        shrinkage_factor = n / (2 * self.system.domain.axis[0].widths()) * rate
        # Add contribution to birth and death
        birth = tf.concat([shrinkage_factor[1:], tf.zeros(1, dtype=tf.float32)], axis=0)
        death = shrinkage_factor
        return birth - death

    def agglomeration(self, n, rate):
        # Transform rate to rate-matrix
        rate_upper_triangular = tfp.math.fill_triangular(rate, upper=True)
        # Agglomeration calculations
        delta_function = tf.constant(1, dtype=tf.float32) - tf.constant(0.5) * tf.eye(self.system.domain.axis[0].m)
        frequency_matrix = rate_upper_triangular * tf.einsum('i,k->ik', n, n)/tf.reduce_sum(n)
        # Calculate birth and death
        #birth = tf.einsum('ijk,jk->i', self.agglomeration_contribution, delta_function*frequency_matrix) #delta_function
        birth = tf.einsum('ijk,jk->i', self.agglomeration_contribution, delta_function*frequency_matrix)
        death = n*tf.einsum('k,ik->i', n, rate_upper_triangular+tf.transpose(rate_upper_triangular*(tf.constant(1, dtype=tf.float32)-tf.eye(self.system.domain.axis[0].m))))/tf.reduce_sum(n)
        #death = tf.einsum('ik->i', frequency_matrix)
        return birth - death

    def agglomeration_constant(self):
        # Initialize constant
        constant = np.zeros([self.system.domain.axis[0].m, self.system.domain.axis[0].m, self.system.domain.axis[0].m])
        # Run agglomeration constant loop
        for i in range(self.system.domain.axis[0].m):
            for j in range(self.system.domain.axis[0].m):
                for k in range(self.system.domain.axis[0].m):
                    new_volume = 1/6*np.pi*self.system.domain.axis[0].midpoints()[j] ** 3 + \
                                 1/6*np.pi*self.system.domain.axis[0].midpoints()[k] ** 3
                    boundary_1 = 1/6*np.pi*self.system.domain.axis[0].midpoints()[i] ** 3
                    if i < self.system.domain.axis[0].m - 1:
                        boundary_2 = 1/6*np.pi*self.system.domain.axis[0].midpoints()[i + 1] ** 3
                    # If new_volume > max_volume
                    if new_volume > 1/6*np.pi*self.system.domain.axis[0].midpoints()[-1] ** 3:
                        constant[-1, j, k] = new_volume / (1/6*np.pi*self.system.domain.axis[0].midpoints()[-1] ** 3)
                    elif boundary_1 <= new_volume <= boundary_2 and i < self.system.domain.axis[0].m - 1:
                        constant[i, j, k] = (boundary_2 - new_volume) / (boundary_2 - boundary_1)
                        constant[i + 1, j, k] = 1 - constant[i, j, k]
        return tf.convert_to_tensor(constant, dtype=tf.float32)

    def breakage(self, n, rate):
        # Divide rate into rate of breakage and breakage distribution parameters
        [breakage_distribution,
         frequency_rate] = tf.split(rate, [round(self.system.domain.axis[0].m*(self.system.domain.axis[0].m-1)/2),
                                           self.system.domain.axis[0].m], 0)
        # Generate temporary breakage matrix
        temp_breakage_matrix = tfp.math.fill_triangular(breakage_distribution)
        # Padding to matrix (adding zero-padding at top and right part of matrix
        padding = tf.constant([[1, 0, ], [0, 1]])
        breakage_matrix = tf.pad(temp_breakage_matrix, padding, "CONSTANT")
        # Add unity to top left of matrix (to avoid division with zero)
        diagonal = tf.concat([tf.ones(1, dtype=tf.float32), tf.zeros(self.system.domain.axis[0].m-1,
                                                                     dtype=tf.float32)], axis=0)
        breakage_matrix = tf.linalg.set_diag(breakage_matrix, diagonal)
        # Normalize breakage matrix
        breakage_matrix = breakage_matrix / tf.reshape(tf.reduce_sum(breakage_matrix, axis=1), (-1, 1))
        # Calculate volume of particles
        particle_volume = 1/6*np.pi*self.system.domain.axis[0].midpoints()**3
        # Calculate number of daughter particles
        daughter_particles = particle_volume/(tf.reduce_sum(breakage_matrix*particle_volume, axis=1))
        # Calculate birth and death
        birth = tf.einsum('ki,k->i', breakage_matrix, daughter_particles*frequency_rate*n)
        death = frequency_rate*n
        return birth - death

    def output_shape(self, input_shape):
        disc_size = input_shape[0][1]
        process_variables = input_shape[1][1]
        return [(None, disc_size), (None, process_variables)]

    def create_pbm_model(self, distribution_dimension: int,
                         process_variable_dimension: int, rate_sizes: Dict[str, int], name: str) -> Model:

        if sum(rate_sizes.values()) < 1:
            raise ValueError('No phenomena activated in PopulationBalanceModel - calculations have been stopped')

        # Get input vectors
        initial_distribution = Input(shape=(distribution_dimension,), name='Initial_distribution')
        process_variables = Input(shape=(2, process_variable_dimension,), name='Process_variables')
        time = Input(shape=(1,), name='Time')
        nucleation_rates = Input(shape=(rate_sizes['nucleation'],), name='Nucleation_rates')
        growth_rates = Input(shape=(rate_sizes['growth'],), name='Growth_rates')
        shrinkage_rates = Input(shape=(rate_sizes['shrinkage'],), name='Shrinkage_rates')
        agglomeration_rates = Input(shape=(rate_sizes['agglomeration'],), name='Agglomeration_rates')
        breakage_rates = Input(shape=(rate_sizes['breakage'],), name='Breakage_rates')
        # Set up layer
        z_1 = Lambda(self.batch, output_shape=self.output_shape,
                     name='Population_balance_model')([initial_distribution, process_variables, time,
                                                       nucleation_rates, growth_rates,
                                                       shrinkage_rates, agglomeration_rates,
                                                       breakage_rates])
        # z_1 = Lambda(self.parallelized_solver, output_shape=self.output_shape,
        #              name='Population_balance_model')([initial_distribution, process_variables, time,
        #                                                nucleation_rates, growth_rates,
        #                                                shrinkage_rates, agglomeration_rates,
        #                                                breakage_rates])
        # Divide output
        predicted_distribution = z_1[0]
        predicted_pv = z_1[1]
        # Generate PBM model
        pbm_model = Model(inputs=[initial_distribution, process_variables, time, nucleation_rates,
                                  growth_rates, shrinkage_rates, agglomeration_rates, breakage_rates],
                          outputs=[predicted_distribution, predicted_pv], name=name)
        #pbm_model.summary()
        return pbm_model
