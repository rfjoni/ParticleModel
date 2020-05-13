from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
import numpy as np
from hybrid_model.HybridModel import HybridModel
from hybrid_model.TimeSeriesPair import TimeSeriesPair
from data.Data import Data
from hybrid_model.Prediction import Prediction
import os


class Illustration:
    def __init__(self, domain):
        self.domain = domain
        self.plot_size = (4*1.2, 3*1.2)
        self.dpi = 400

    def plot_loss(self, training: History, ref_loss_train: dict = None, ref_loss_val: dict = None) -> Figure:
        fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
        sns.lineplot(x=[epoch + 1 for epoch in training.epoch], y=training.history['loss'], ax=ax, label='Training',
                     color=sns.color_palette()[0])
        sns.lineplot(x=[epoch + 1 for epoch in training.epoch], y=training.history['val_loss'], ax=ax,
                     label='Validation', color=sns.color_palette()[1])
        if ref_loss_train:
            ax.plot([epoch + 1 for epoch in training.epoch],
                    [ref_loss_train['Average loss'] for _ in training.epoch], '--', label='Reference training',
                    color=sns.color_palette()[0])
        if ref_loss_val:
            ax.plot([epoch + 1 for epoch in training.epoch],
                    [ref_loss_val['Average loss'] for _ in training.epoch], '--',
                    label='Reference validation', color=sns.color_palette()[1])
        ax.legend()
        ax.set_xlabel('Epochs [-]')
        ax.set_ylabel('Loss [-]')
        plt.show()
        return fig

    def plot_explained_variation(self, history: dict, ref_loss_train: dict, ref_loss_val: dict,
                                 history_2: dict = None) -> Figure:
        fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)

        def plot_explained_variation(history_dict, label_extra, ax):
            training_accuracy = [(ref_loss_train['Average loss']-loss)/ref_loss_train['Average loss']*100
                                 for loss in history_dict['loss']]
            validation_accuracy = [(ref_loss_val['Average loss']-loss)/ref_loss_val['Average loss']*100 for
                                   loss in history_dict['val_loss']]
            sns.lineplot(x=[epoch + 1 for epoch in range(len(history_dict['loss']))], y=training_accuracy, ax=ax, label='Training'+label_extra)
            sns.lineplot(x=[epoch + 1 for epoch in range(len(history_dict['val_loss']))], y=validation_accuracy, ax=ax, label='Validation'+label_extra)
        if history_2 is not None:
            for history_dict, label_extra in zip([history, history_2], [' (regularization)', ' (no regularization)']):
                plot_explained_variation(history_dict, label_extra, ax=ax)
        else:
            plot_explained_variation(history, label_extra='', ax=ax)
        ax.legend()
        ax.set_xlabel('Epochs [-]')
        ax.set_ylabel('Accuracy [%]')
        plt.show()
        return fig

    def plot_multi_prediction(self, hybrid_model: HybridModel, prediction_data: dict,
                              start_step: list, target_step: int, relative=False, reference_data = None) -> Figure:
        # Prepare prediction data
        [x_pred, y_pred] = hybrid_model.model_data(prediction_data)
        fig, ax = plt.subplots(figsize=self.plot_size, dpi=self.dpi)
        for init_step in start_step:
            # Set multi prediction inputs from data-set
            initial_distribution = np.expand_dims(x_pred[0][init_step], axis=0)
            initial_dilution_factor = np.expand_dims(x_pred[5][init_step], axis=0)
            initial_process_variables = np.expand_dims(x_pred[1][init_step], axis=0)
            target_distribution = np.expand_dims(y_pred[target_step-1], axis=0)
            target_dilution_factor = np.expand_dims(x_pred[6][target_step-1], axis=0)
            controlled_variables = np.expand_dims(x_pred[2][init_step:target_step], axis=0)
            time_steps = np.expand_dims(x_pred[7][init_step:target_step], axis=0)
            model_input = [initial_distribution,
                           initial_dilution_factor,
                           initial_process_variables,
                           target_distribution,
                           target_dilution_factor,
                           controlled_variables,
                           time_steps]

            output = hybrid_model.multi_step_prediction(steps=target_step - init_step).predict(x=model_input)

            print('Mass initial: '+str(np.sum(output[2][0]*1/6*np.pi*hybrid_model.system.domain.axis[0].midpoints()**3)))
            print('Mass prediction: ' + str(np.sum(output[0][0] * 1 / 6 * np.pi * hybrid_model.system.domain.axis[0].midpoints() ** 3)))

            pred = output[0][0]
            #pred = pred / self.domain.axis[0].widths()
            if relative:
                pred = pred*np.pi/6*self.domain.axis[0].midpoints()**3/np.sum(pred*np.pi/6*self.domain.axis[0].midpoints()**3)
                pred = pred / self.domain.axis[0].widths()
                pred = pred / np.sum(pred)
            ref = output[1][0]
            #ref = ref / self.domain.axis[0].widths()
            if relative:
                ref = ref*np.pi/6*self.domain.axis[0].midpoints()**3 / np.sum(ref*np.pi/6*self.domain.axis[0].midpoints()**3)
                ref = ref / self.domain.axis[0].widths()
                ref = ref / np.sum(ref)
            sns.lineplot(x=self.domain.axis[0].midpoints(), y=pred, ax=ax,
                         label='Prediction at t = ' + str(np.round(np.sum(x_pred[7][0:init_step]) / 60 / 60, 1)) + ' hrs')
            if relative:
                print('Prediction at t = ' + str(np.round(np.sum(x_pred[7][0:init_step]) / 60 / 60, 1)) + ' hrs, Error: ' + str(np.sum(np.abs(pred-ref)/2*100)))
            else:
                print('Prediction at t = ' + str(np.round(np.sum(x_pred[7][0:init_step]) / 60 / 60, 1)) + ' hrs, Error: ' + str(np.sum(np.abs(pred-ref))))

        if reference_data is not None:
            #reference_data = reference_data / self.domain.axis[0].widths()
            if relative:
                reference_data = reference_data*np.pi/6*self.domain.axis[0].midpoints()**3 / np.sum(reference_data*np.pi/6*self.domain.axis[0].midpoints()**3)
                reference_data = reference_data/self.domain.axis[0].widths()
                reference_data = reference_data/np.sum(reference_data)
            sns.lineplot(x=self.domain.axis[0].midpoints(), y=reference_data, ax=ax,
                         label='Reference prediction at t = 0.0 hrs')

        sns.lineplot(x=self.domain.axis[0].midpoints(), y=ref, ax=ax,
                     label='Measured at t = ' + str(np.round(np.sum(x_pred[7][0:target_step]) / 60 / 60, 1)) + ' hrs',
                     color='black')
        ax.set_xlabel('Particle size, '+self.domain.axis[0].disc_by+' [µm]')
        ax.lines[-1].set_linestyle("--")
        if relative:
            ax.set_ylabel('Relative volume density [-]')
        else:
            ax.set_ylabel('Absolute number density [-/µL]')
        leg = ax.legend()
        leg_lines = leg.get_lines()
        leg_lines[-1].set_linestyle("--")
        plt.show()
        return fig


