from data.Data import Data, Batch
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
from hybrid_model.System import Domain
colors = ['#4c72b0',  '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd',
          '#000000', '#808080']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))
plt.ion()


class Illustration:
    def __init__(self, data: Data):
        self.data = data
        #self.plot_size = (4*1.2, 3*1.2)
        self.plot_size = (4*1, 3*1)
        self.dpi = 400

    def data_to_lists(self, batch_ids: list = None) -> List[Dict[str, Dict[str, List[Union[float, np.ndarray]]]]]:
        # List of sensors
        external_sensors_list = {}
        particle_analysis_sensors_list = {}
        image_sensors_list = {}
        # List of batches
        if batch_ids:
            batch_list = list(self.data.batches[i] for i in batch_ids)
        else:
            batch_list = self.data.batches
        # Check all available sensor readings in measurements
        for batch in batch_list:
            # Initialize for batch
            external_sensors_list[batch.batch_id] = {}
            particle_analysis_sensors_list[batch.batch_id] = {}
            image_sensors_list[batch.batch_id] = {}
            # Loop for each measurement in batch
            for measurement in batch.measurements:
                for sensor_category, sensors in zip([external_sensors_list[batch.batch_id],
                                                     particle_analysis_sensors_list[batch.batch_id],
                                                     image_sensors_list[batch.batch_id]],
                                                    [measurement.external_sensors,
                                                     measurement.particle_analysis_sensors,
                                                     measurement.image_sensors]):
                    for sensor in sensors:
                        if sensor.sensor_id not in sensor_category.keys():
                            sensor_category[sensor.sensor_id] = {'time': [],
                                                                 'value': [],
                                                                 'unit': []}
                        sensor_category[sensor.sensor_id]['time'].append(measurement.time)
                        sensor_category[sensor.sensor_id]['value'].append(sensor.value)
                        sensor_category[sensor.sensor_id]['unit'].append(sensor.unit)
        return [external_sensors_list, particle_analysis_sensors_list, image_sensors_list]

    def plot_variable(self, variable_name: str, axis_label: str, category: str, batch_ids: list = None):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        if category == 'external':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            for batch_id, sensors in external_data.items():
                start_time = mdates.date2num(min(sensors[variable_name]['time']))
                time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name]['time'])]
                ax.plot(time, sensors[variable_name]['value'], label=batch_id)
            #ax.legend(loc='upper right')
            ax.legend()
            ax.grid()
            ax.set_xlabel('Time [hh:mm]')
            ax.set_ylabel(axis_label)
            ax.set_xlim([1, None])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.xticks(rotation=-45)
            return fig

        if category == 'image':
            fig, ax = plt.subplots()
            for batch_id, sensors in image_data.items():
                ax.imshow(sensors[variable_name]['value'][0], cmap='gray', vmin=0, vmax=255)
            return fig

        if category == 'particle_analysis':
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 4.5), dpi=self.dpi)
            ax1.plot()
            ax2.plot()
            ax3.plot()
            for batch_id, sensors in particle_analysis.items():
                start_time = mdates.date2num(min(sensors[variable_name]['time']))
                time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name]['time'])]
                d10 = [np.percentile(particle_list, q=10) for particle_list in sensors[variable_name]['value']]
                d50 = [np.percentile(particle_list, q=50) for particle_list in sensors[variable_name]['value']]
                d90 = [np.percentile(particle_list, q=90) for particle_list in sensors[variable_name]['value']]
                ax1.plot(time, d10, label=batch_id)
                ax2.plot(time, d50, label=batch_id)
                ax3.plot(time, d90, label=batch_id)
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.legend()
            ax2.legend()
            ax3.legend()
            #ax3.legend(loc='upper left', fontsize='small')
            for ax, ax_type in zip([ax1, ax2, ax3],['D10', 'D50', 'D90']):
                ax.set_xlabel('Time [hh:mm]')
                ax.set_ylabel(ax_type + ' ' + axis_label)
                ax.set_xlim([1, None])
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                for label in ax.get_xticklabels():
                    label.set_rotation(-45)
            return fig

    def plot_variable_batch(self, variable_name: str, batch: str, axis_label: str, category: str):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists()
        if category == 'particle_analysis':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            ax.plot()
            for batch_id, sensors in particle_analysis.items():
                if batch_id==batch:
                    start_time = mdates.date2num(min(sensors[variable_name]['time']))
                    time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name]['time'])]
                    for measurement_id, particle_list in enumerate(sensors[variable_name]['value']):
                        sns.distplot(particle_list, ax=ax, label=time[measurement_id])
                #ax.legend()
                #ax.set_xscale('log')
                ax.set_xlim([0.5, 25])
                ax.set_xlabel(axis_label)
                ax.set_xlabel('Relative particle density [-]')
                # ax.set_xlabel('Time [hh:mm]')
                # ax.set_ylabel(ax_type + ' ' + axis_label)
                # ax.xaxis_date()
                # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                # ax.set_xlim([1, None])
            plt.xticks(rotation=-45)
            return fig

    def plot_time_distribution(self, variable_name: str, axis_label: str, category: str, time_step: int, batch_ids: list = None):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        if category == 'particle_analysis':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            ax.plot()
            for batch_id, sensors in particle_analysis.items():
                sns.distplot(sensors[variable_name]['value'][time_step], ax=ax, label=batch_id, hist=False)
            ax.grid()
            ax.legend()
            ax.set_xlim([0, None])
            #ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel(axis_label)
            ax.set_ylabel('Normalized count [-]')
            plt.xticks(rotation=-45)
            return fig

    def plot_distribution(self, variable_name: str, axis_label: str, category: str, start: int, stop: int, samples: int, domain: Domain, batch_ids: list = None):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        if category == 'particle_analysis':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            ax.plot()
            for batch_id, sensors in particle_analysis.items():
                for time_step in range(start, stop, round((stop-start)/samples)):
                    counts, bins = np.histogram(sensors[variable_name]['value'][time_step], density=True, bins=domain.axis[0].edges()) #weights=sensors[variable_name]['value'][time_step]**3
                    norm_count = counts/domain.axis[0].widths()/np.sum(counts/domain.axis[0].widths())
                    sns.lineplot(x=domain.axis[0].midpoints(), y=norm_count, label='t = '+str((sensors[variable_name]['time'][time_step]-sensors[variable_name]['time'][0]))[7:-3])
                    #sns.distplot(sensors[variable_name]['value'][time_step]**3, ax=ax, label=str(sensors[variable_name]['time'][time_step]), hist=False)
            ax.grid()
            ax.legend()
            ax.set_xlim([0, None])
            #ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel(axis_label)
            ax.set_ylabel('Relative volume density [-]')
            plt.xticks(rotation=-45)
            return fig

    def plot_count_time(self, variable_name: str, category: str, batch_ids: list = None, lb: float = 0, ub: float = np.inf,):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        if category == 'particle_analysis':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            ax.plot()
            for batch_id, sensors in particle_analysis.items():
                start_time = mdates.date2num(min(sensors[variable_name]['time']))
                time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name]['time'])]
                count = [len(particle_list[(particle_list >= lb) & (particle_list <= ub)]) for particle_list in sensors[variable_name]['value']]
                ax.plot(time, count, label=batch_id)
            ax.grid()
            ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel('Time [hh:mm]')
            ax.set_ylabel('Count [-]')
            ax.set_xlim([1, None])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.xticks(rotation=-45)
            return fig

    def plot_fractile_time(self, variable_name: str, category: str, fractile: float, axis_label: str,
                           lb: float = 0, ub: float = np.inf, batch_ids: list = None):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        if category == 'particle_analysis':
            fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
            for batch_id, sensors in particle_analysis.items():
                start_time = mdates.date2num(min(sensors[variable_name]['time']))
                time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name]['time'])]
                d = []
                for p_list in sensors[variable_name]['value']:
                    filtered_list = p_list[(p_list >= lb) & (p_list <= ub)]
                    if len(filtered_list) == 0:
                        d.append(0)
                    else:
                        d.append(np.percentile(filtered_list, q=fractile))
                ax.plot(time, d, label=batch_id)
            ax.grid()
            ax.legend()
            ax.set_xlabel('Time [hh:mm]')
            ax.set_ylabel('D' + str(fractile) + ' ' + axis_label)
            ax.set_xlim([1, None])
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.xticks(rotation=-45)
            return fig

    def plot_dual_axis(self, variable_name_1: str, category_1: str, axis_label_1: str,
                       variable_name_2: str, category_2: str, axis_label_2: str,
                       fractile: float, lb: float = 0, ub: float = np.inf, batch_ids: list = None):
        # Call data-set
        [external_data, particle_analysis, image_data] = self.data_to_lists(batch_ids)
        fig, ax_y1 = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
        color_y1 = 'black'
        color_y2 = (237/255,125/255,49/255)
        line_style = ['-', '--']
        if category_1 == 'external':
            for style, (batch_id, sensors) in enumerate(external_data.items()):
                start_time = mdates.date2num(min(sensors[variable_name_1]['time']))
                time = [(time - start_time) + 1 for time in mdates.date2num(sensors[variable_name_1]['time'])]
                ax_y1.plot(time, sensors[variable_name_1]['value'], label=batch_id, color=color_y1, linestyle=line_style[style])
        ax_y2 = ax_y1.twinx()
        if category_2 == 'particle_analysis':
            for style, (batch_id, sensors) in enumerate(particle_analysis.items()):
                start_time = mdates.date2num(min(sensors[variable_name_2]['time']))
                time = [(time - start_time)+1 for time in mdates.date2num(sensors[variable_name_2]['time'])]
                d = [np.percentile(particle_list[(particle_list >= lb) & (particle_list <= ub)],
                                   q=fractile) for particle_list in sensors[variable_name_2]['value']]
                ax_y2.plot(time, d, label=batch_id, color=color_y2, linestyle=line_style[style])
        ax_y1.legend(loc='upper left')
        ax_y1.grid()
        ax_y1.set_xlabel('Time [hh:mm]')
        ax_y1.xaxis_date()
        ax_y1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_y1.set_xlim([1, None])
        ax_y1.set_ylabel(axis_label_1, color=color_y1)
        ax_y1.tick_params(axis='y', labelcolor=color_y1)
        ax_y2.set_ylabel(axis_label_2, color=color_y2)
        ax_y2.tick_params(axis='y', labelcolor=color_y2)
        for label in ax_y1.get_xticklabels():
            label.set_rotation(-45)
        return fig
