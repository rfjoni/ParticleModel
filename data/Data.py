# -*- coding: utf-8 -*-
"""Data structure

This module contains universal data structure objects for storage of time-series data

"""

# Importing dependencies
import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
from typing import List
import datetime
import numpy as np


class Sensor:
    """Class for single sensor input

    Attributes:
        sensor_id: Name of sensor
        value: Measurement value
        data_type: Data type (single_value or distribution)
        std_error: Standard error of measurement

    """

    def __init__(self, sensor_id: str, value: [float, np.ndarray], data_type: str,
                 std_error: [float, np.ndarray] = None, unit: str = None) -> None:
        """
        Creating sensor object

        Args:
            sensor_id: Name of sensor
            value: Sensor value
            data_type: Type of data
            std_error: Standard error of measurement
        """
        self.sensor_id = sensor_id
        self.value = value
        self.data_type = data_type
        self.std_error = std_error
        self.unit = unit


class Measurement:
    """Class for single measurement

    Attributes:
        measurement_id: Measurement name
        external_sensors: List of external sensor inputs
        particle_analysis_sensors: List of particle analysis sensor inputs
        time: Time of measurement in datetime format
    """

    def __init__(self, measurement_id: str, time: datetime.datetime) -> None:
        """
        Creating measurement object

        Args:
            measurement_id: Name of measurement
            time: Time of measurement (datetime object)
        """
        self.measurement_id = measurement_id
        self.external_sensors: List[Sensor] = []
        self.particle_analysis_sensors: List[Sensor] = []
        self.image_sensors: List[Sensor] = []
        self.time = time

    def add_external_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of external sensors

        Args:
            sensor: Sensor object
        """
        self.external_sensors.append(sensor)

    def add_particle_analysis_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of particle analysis sensors

        Args:
            sensor: Sensor object
        """
        self.particle_analysis_sensors.append(sensor)

    def add_image_sensor(self, sensor: Sensor) -> None:
        """
        Add sensor object to list of image sensors

        Args:
            sensor: Sensor object
        """
        self.image_sensors.append(sensor)


class Batch:
    """Class for batch data

    Attributes:
        batch_id: Name of batch (use unique naming)
        measurements: List of measurement objects
        pool: Dictionary of data pool types
    """

    def __init__(self, batch_id: str) -> None:
        """
        Creating batch object

        Args:
            batch_id: Name of batch
        """
        self.batch_id = batch_id
        self.measurements: List[Measurement] = []
        self.pool = dict()

    def add_measurement(self, measurement: Measurement) -> None:
        """
        Add measurement object to list of measurements

        Args:
            measurement: Measurement object
        """
        self.measurements.append(measurement)

    def add_measurment_from_text(self, time, temperature, density, density_temperature, image_analysis_data):
        measurement_object = Measurement(measurement_id='Manual measurement', time=time)
        # Get list of image analysis sensors
        list_of_image_analysis_sensors = image_analysis_data.columns.values.tolist()
        for image_analysis_sensor_id in list_of_image_analysis_sensors:
            value = image_analysis_data.loc[:, image_analysis_sensor_id].values
            particle_analysis_sensor_object = Sensor(image_analysis_sensor_id, value, 'image_analysis')
            measurement_object.add_particle_analysis_sensor(particle_analysis_sensor_object)
        # Add other sensors
        for sensor_reading, external_sensor_id in zip([temperature, density, density_temperature], ['Temperature','Density','Density_temperature']):
            value = sensor_reading
            external_sensor_object = Sensor(external_sensor_id, value, 'external')
            measurement_object.add_external_sensor(external_sensor_object)
        # Add concentration sensor
        conc_calibration = np.array([3.14794, -0.010415, 0.0201494,
                                     6.24689e-06, -0.000299322, 1.58896e-05])
        def concentration_model(coef, temp, dens):
            dens = dens*1000
            input_matrix = np.array([1,
                                     dens,
                                     temp,
                                     dens ** 2,
                                     temp ** 2,
                                     dens * temp]).T
            out = np.sum(coef * input_matrix)
            return out
        external_sensor_object = Sensor('Concentration', concentration_model(conc_calibration,
                                                                             density_temperature,
                                                                             density), 'external')
        measurement_object.add_external_sensor(external_sensor_object)
        self.add_measurement(measurement_object)


class Data:
    """Class containing all data

    Attributes:
        batches: List of batches
        case_id: Case name
    """

    def __init__(self, case_id: str) -> None:
        """
        Creating data object

        Args:
            case_id: Case name
        """
        self.batches: List[Batch] = []
        self.case_id = case_id

    def add_batch(self, batch: Batch) -> None:
        """
        Add batch object to list of batches

        Args:
            batch: Batch object
        """
        self.batches.append(batch)

    def load_from_folder(self, folder_path: str, external_file_name: str, feature_file_name: str,
                         image_file_name: str = None, sub_folder_structure: str = '') -> None:
        """
        Load data from folder structure

        Folder structure as following:
        ##############################
        folder_path/
            - external_file_name
            - batch_id/
                - measurement_id/
                    - feature_file_name
                    - image_file_name
            ...
        ##############################

        Args:
            folder_path: Path to data folder
            external_file_name: External data file name (including file extension)
            feature_file_name: Image analysis feature file name (including file extension)
            image_file_name: Image file name (not mandatory) (including file extension)
            sub_folder_structure: Sub folder structure if the feature and image files are not in the root folder
        """
        # Load external data sheet
        external_data = pd.ExcelFile(folder_path + '/' + external_file_name)
        # Get list of batch ids from external data sheets
        list_of_batch_id = external_data.sheet_names

        for batch_id in tqdm(list_of_batch_id, desc='Batches'):
            # Load batch data from sheet corresponding to batch and set folder no. as index
            batch_data = external_data.parse(batch_id).set_index("Folder no.")
            # Create batch object
            batch_object = Batch(batch_id)
            # Get list of measurement ids from index values
            list_of_measurement_id = batch_data.index.values.tolist()


            for measurement_id in tqdm(list_of_measurement_id, desc='Measurements'):
                # Load external sensor measurement data by finding corresponding row
                external_sensor_measurement_data = batch_data.loc[measurement_id]
                # Extract time from data sheet
                time = external_sensor_measurement_data.at['Timestamp']
                # Create measurement object
                measurement_object = Measurement(measurement_id, time)

                # Get list of external sensors
                list_of_external_sensors = batch_data.columns.values.tolist()
                for external_sensor_id in list_of_external_sensors:
                    value = external_sensor_measurement_data.at[external_sensor_id]
                    external_sensor_object = Sensor(external_sensor_id, value, 'external')
                    measurement_object.add_external_sensor(external_sensor_object)

                # Generate image analysis file location
                image_analysis_file = folder_path + '/' + batch_id + '/' +\
                                      str(measurement_id) + '/' + sub_folder_structure + feature_file_name
                # Load image analysis data
                image_analysis_data = pd.ExcelFile(image_analysis_file).parse('Features').set_index("Object.Id")
                # Get list of image analysis sensors
                list_of_image_analysis_sensors = image_analysis_data.columns.values.tolist()
                for image_analysis_sensor_id in list_of_image_analysis_sensors:
                    value = image_analysis_data.loc[:, image_analysis_sensor_id].values
                    particle_analysis_sensor_object = Sensor(image_analysis_sensor_id, value, 'image_analysis')
                    measurement_object.add_particle_analysis_sensor(particle_analysis_sensor_object)

                # Generate image file location
                image_file = folder_path + '/' + batch_id + '/' + str(measurement_id) + '/' +\
                             sub_folder_structure + image_file_name
                # Load image
                image_file_data = Image.open(image_file)
                image_file_data.thumbnail([512, 512])
                image_array_data = np.array(image_file_data)
                image_sensor_object = Sensor('Image_sensor', image_array_data, 'image')
                measurement_object.add_image_sensor(image_sensor_object)

                # Add measurement object to batch object
                batch_object.add_measurement(measurement_object)
            # Add batch object to data object
            self.add_batch(batch_object)

    def add_sensor_uncertainty(self, sensor_id: str, std_error: any) -> None:
        """
        Set measurement uncertainty of given sensor_id in all batches and measurements

        Args:
            sensor_id: Sensor name
            std_error: Constant standard error of sensor
        """
        for batch in self.batches:
            for measurement in batch.measurements:
                all_sensors = measurement.external_sensors + measurement.particle_analysis_sensors + \
                              measurement.image_sensors
                for sensor in all_sensors:
                    if sensor.sensor_id == sensor_id:
                        sensor.std_error = std_error

    def set_batch_pool(self, pool_batch_id: List[str], pool_type: str) -> None:
        """
        Set batch_id to data pool (overrides previous pool_type)

        Args:
            pool_batch_id: List of batch names to set in pool_type
            pool_type: Pool type (training, validation, test, etc.)
        """
        for batch_id in pool_batch_id:
            for batch in self.batches:
                if batch.batch_id == batch_id:
                    # Reset pool definition
                    batch.pool = dict()
                    batch.pool[pool_type] = True
                    print(str(batch_id) + " set to " + str(pool_type) + " data pool")
                    break

    def reset_batch_pools(self):
        for batch in self.batches:
            # Reset pool definition
            batch.pool = dict()
        print("Batch pool(s) have been reset")

    def save_to_pickle(self, filename: str) -> None:
        """
        Save data to pickle file structure

        Args:
            filename: Filename of saved pickle file (without file extension)
        """
        filename = filename or self.case_id
        with open(filename + '.pkl', 'wb') as saved:
            pickle.dump(self.batches, saved, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, filename: str) -> None:
        """
        Loads data from pickle file structure

        Args:
            filename: Filename of pickle file to be loaded (without .pkl) (optional)
        """
        filename = filename or self.case_id
        with open(filename + '.pkl', 'rb') as loaded:
            self.batches.extend(pickle.load(loaded))
