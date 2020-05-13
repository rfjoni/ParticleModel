# -*- coding: utf-8 -*-
"""Time series pair module

This module converts a data-object to pairs of time-series data points, used for training hybrid model

"""

# Importing dependencies
from scipy.stats import binned_statistic
import numpy as np
from data.Data import Data, Sensor
from hybrid_model.System import System
from typing import List, Dict


class TimeSeriesPair:
    """Class containing all data

    Attributes:
        data: Data object
        system: System object
    """

    def __init__(self, data: Data, system: System) -> None:
        """
        Creating training data object

        Args:
            data: Data object
            system: System object
        """
        self.data = data
        self.system = system

    @staticmethod
    def bin_statistics(image_analysis_sensors: list, axes: list, statistic: str = 'count') -> np.ndarray:
        """
        Generating bin statistics from image analysis sensor

        Args:
            image_analysis_sensors: List of image analysis sensors
            axes: List of axes to generate bin statistics from
            statistic: Statistic type

        Returns:
            Bin statistics
        """

        # Get number of dimensions
        number_of_dimensions = len(axes)

        # Initialize axes index vector
        axes_index = []

        # Find sensors corresponding to axis
        for index, axis in enumerate(axes):
            for sensor_index, sensor in enumerate(image_analysis_sensors):
                if sensor.sensor_id == axis.disc_by:
                    axes_index.append(int(sensor_index))

        # Stack samples from specified axes
        sample_stack = np.squeeze(np.stack([image_analysis_sensors[index].value for index in axes_index]))
        # Stack discretizations
        bins = np.squeeze(np.stack([axes[index].edges() for index in range(number_of_dimensions)]))
        # Calculate binned statistic
        if sample_stack.any():
            bin_result, _, _ = binned_statistic(sample_stack, sample_stack, statistic=statistic, bins=bins)
        else:
            bin_result = np.zeros_like(
                np.squeeze(np.stack([axes[index].midpoints() for index in range(number_of_dimensions)])))
        return bin_result

    @staticmethod
    def concentration_correction(external_sensors: List[Sensor]) -> np.ndarray:
        """
        Correction factor for any dilution of samples

        Args:
            external_sensors: list of external sensors

        Returns:
            Vector of concentration correction factors
        """

        # Initialize sample and dilution volumes, in case no dilution
        sample_volume = 1
        dilution_volume = 0
        flowcell_volume = 1

        for sensor in external_sensors:
            if 'Sample' in sensor.sensor_id:
                sample_volume = sensor.value
            if 'Water' in sensor.sensor_id:
                dilution_volume = sensor.value
            if 'Flowcell volume' in sensor.sensor_id:
                flowcell_volume = sensor.value

        correction_factor = sample_volume / (sample_volume + dilution_volume)
        return np.array([correction_factor, flowcell_volume])

    def shuffle(self, pool_type: List[str], start: int = 0, stop: int = np.inf, min_step: int = 1,
                max_step: int = np.inf, delta_t_critical: float = None, delta_t_min: float = None) -> Dict[str, np.ndarray]:
        """
        Shuffle data points

        Args:
            pool_type: Pool type
            start: Start index for shuffling
            stop: Stop index for shuffling
            min_step: Minimum step in shuffling
            max_step: Maximum step in shuffling
            delta_t_critical: Critical sampling time
        Returns:
            Dictionary of training data
        """

        # Initialize output
        current_distribution = []
        future_distribution = []
        measured_variable = []
        measured_variable_uncertainty = []
        controlled_variable = []
        controlled_variable_uncertainty = []
        delta_time = []
        current_image = []
        current_dilution_correction = []
        future_dilution_correction = []

        # Shuffle input data
        for batch in self.data.batches:
            for pool in pool_type:
                if pool in batch.pool and batch.pool[pool]:
                    # Number of measurement
                    number_of_measurements = len(batch.measurements)
                    # Define shuffle parameters
                    shuffle_end = min(number_of_measurements-1, stop)
                    shuffle_start = max(0, min(start, shuffle_end - min_step))
                    for start_index in range(shuffle_start, shuffle_end):
                        for end_index in range(start_index + min_step,
                                               min(start_index + max_step, shuffle_end)+1):
                            # Load measurements for start and end indexes
                            start_measurement = batch.measurements[start_index]
                            end_measurement = batch.measurements[end_index]
                            # Delta time
                            delta_time_test = (end_measurement.time - start_measurement.time).total_seconds()

                            # Check if delta_t_critical and delta_t_min is violated
                            if delta_t_critical is not None and delta_t_critical < delta_time_test:
                                # Jump to next iteration in the loop
                                continue
                            if delta_t_min is not None and delta_t_min > delta_time_test:
                                # Jump to next iteration in the loop
                                continue

                            delta_time.append(delta_time_test)
                            # Modelled distribution
                            current_distribution.append(
                                self.bin_statistics(image_analysis_sensors=start_measurement.particle_analysis_sensors,
                                                    axes=self.system.domain.axis))
                            future_distribution.append(
                                self.bin_statistics(image_analysis_sensors=end_measurement.particle_analysis_sensors,
                                                    axes=self.system.domain.axis))

                            # Measured and controlled variables
                            temp_measured = []
                            temp_controlled = []
                            temp_measured_uncertainty = []
                            temp_controlled_uncertainty = []
                            for sensor in self.system.sensors:
                                for data_sensor_now in start_measurement.external_sensors:
                                    # If sensor is selected as measured
                                    if sensor.name == data_sensor_now.sensor_id and sensor.measured:
                                        temp_measured.append(data_sensor_now.value)
                                        temp_measured_uncertainty.append(data_sensor_now.std_error or 0)
                                for data_sensor_future in end_measurement.external_sensors:
                                    # If sensor is selected as controlled
                                    if sensor.name == data_sensor_future.sensor_id and sensor.controlled:
                                        temp_controlled.append(data_sensor_future.value)
                                        temp_controlled_uncertainty.append(data_sensor_future.std_error or 0)
                            # Save variables
                            measured_variable.append(temp_measured)
                            controlled_variable.append(temp_controlled)
                            measured_variable_uncertainty.append(temp_measured_uncertainty)
                            controlled_variable_uncertainty.append(temp_controlled_uncertainty)

                            # Dilution correction
                            current_dilution_correction.append(
                                self.concentration_correction(start_measurement.external_sensors))
                            future_dilution_correction.append(
                                self.concentration_correction(end_measurement.external_sensors))

                            # Current image
                            if start_measurement.image_sensors:
                                current_image.append(start_measurement.image_sensors[0].value)
                            else:
                                current_image.append(np.zeros([64 * 64]))

        return {'Current distribution': np.asarray(current_distribution),
                'Future distribution': np.asarray(future_distribution),
                'Measured variables': np.asarray(measured_variable),
                'Measured variable uncertainty': np.asarray(measured_variable_uncertainty).astype('float32'),
                'Controlled variables': np.asarray(controlled_variable),
                'Controlled variable uncertainty': np.asarray(controlled_variable_uncertainty).astype('float32'),
                'Time horizon': np.expand_dims(np.asarray(delta_time), axis=1),
                'Current image': np.asarray(current_image),
                'Current dilution factor': np.asarray(current_dilution_correction),
                'Future dilution factor': np.asarray(future_dilution_correction)}
