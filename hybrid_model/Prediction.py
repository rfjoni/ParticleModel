import numpy as np


class Prediction:
    def __init__(self, hybrid_model):
        pass

    def multi_step_prediction(self, hybrid_model, data_shuffler, start_step, prediction_step):
        # Initialize
        psd_pred = []
        pv_pred = []
        psd_real = []
        pred_time = []
        rate_pred = []
        psd_current = []
        # Get data
        [x_test, y_test] = self.hybrid_model.model_data(data_shuffler)
        # Initial PV's
        distribution = np.expand_dims(x_test[0][start_step], axis=0)
        dilution_factor = np.expand_dims(x_test[5][start_step], axis=0)
        measured_variables = np.expand_dims(x_test[1][start_step], axis=0)
        PSD = self.hybrid_model.sub_models['Dilution in'].predict([distribution, dilution_factor])

        for time_step in range(start_step, prediction_step):
            # Current PSD
            current_distribution = np.expand_dims(x_test[0][time_step], axis=0)
            current_dilution_factor = np.expand_dims(x_test[5][time_step], axis=0)
            current_psd = self.hybrid_model.sub_models['Dilution in'].predict([current_distribution, current_dilution_factor])
            # Time horizon
            time = np.expand_dims(x_test[7][time_step], axis=0)
            # Local control
            controlled_variables = np.expand_dims(x_test[2][time_step], axis=0)
            # Reference
            target_dilution_factor = np.expand_dims(x_test[6][time_step], axis=0)
            target_distribution = np.expand_dims(y_test[time_step], axis=0)
            target_PSD = self.hybrid_model.sub_models['Dilution in'].predict([target_distribution, target_dilution_factor])
            # Simulate one step
            PV = self.hybrid_model.sub_models['Process states'].predict([measured_variables, controlled_variables, time])
            rate = self.hybrid_model.sub_models['Rate'].predict(x=[PV, PSD])
            pred = self.hybrid_model.sub_models['Population Balance Model Prediction'].predict(x=[PSD, PV, time]+rate)
            # Save intermediates
            psd_current.append(current_psd[0])
            psd_pred.append(pred[0][0])
            pv_pred.append(pred[1][0])
            psd_real.append(target_PSD[0])
            pred_time.append(time[0])
            rate_pred.append([r[0] for r in rate])
            # Save results for next iteration
            PSD = pred[0]
            measured_variables = pred[1]
        return [psd_pred, pv_pred, psd_real, pred_time, rate_pred, psd_current]