from data.Data import Data
from domain.Domain import Domain
import numpy as np
from hybrid_model.TimeSeriesPair import TimeSeriesPair
from scipy.optimize import minimize
from reference_model.Lactose_kinetic_fitting.CrystallizationModel import CrystallizationModel
from reference_model.Lactose_kinetic_fitting.CrystallizationKinetics import CrystallizationKinetics, LactoseSolute, LactoseGrowth, LactoseNucleation
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from Timer import Timer
import lmfit

# Construct discretized domain object for hybrid model
domain = Domain(name='Domain')
domain.add_axis(x_min=5, x_max=100, m=30, disc_by='FeretMean', name='FeretMean')

# Create data-set and set up data-shuffler
data = Data(case_id='Laboratory lactose case study')
data.load_from_pickle('C:/Users/rfjoni/PycharmProjects/ParticleModel/projects/CACE_cases/CACE_lactose_study/lactose')
data.batches[2].batch_id = 'Batch 1'
data.batches[3].batch_id = 'Batch 2'

# Convert time and temperature data to polynomial fit
# Batch 1
t_batch1 = [(measurement.time-data.batches[2].measurements[0].time).total_seconds() for measurement in data.batches[2].measurements]
T_batch1 = [measurement.external_sensors[2].value for measurement in data.batches[2].measurements]
Tfunc_batch1 = np.polyfit(t_batch1, T_batch1, deg=3)
N_batch1 = np.array([(TimeSeriesPair.bin_statistics(measurement.particle_analysis_sensors, axes=domain.axis)/measurement.external_sensors[5].value).tolist() for measurement in data.batches[2].measurements]).T
# Batch 2
t_batch2 = [(measurement.time-data.batches[3].measurements[0].time).total_seconds() for measurement in data.batches[3].measurements]
T_batch2 = [measurement.external_sensors[2].value for measurement in data.batches[3].measurements]
Tfunc_batch2 = np.polyfit(t_batch2, T_batch2, deg=3)
N_batch2 = np.array([(TimeSeriesPair.bin_statistics(measurement.particle_analysis_sensors, axes=domain.axis)/measurement.external_sensors[5].value).tolist() for measurement in data.batches[3].measurements]).T


# Solute model
solute = LactoseSolute()

# Kinetic models
nucleation = LactoseNucleation()
growth = LactoseGrowth(domain=domain)
kinetics = CrystallizationKinetics(nucleation=nucleation, growth=growth)

# Set up model
mechanistic_model = CrystallizationModel(domain=domain, kinetics=kinetics, solute=solute)
out = mechanistic_model.solve_ode(t_steps=t_batch1,
                                  z=Tfunc_batch1,
                                  parameters=[1, 1, 1, 1, 1, 1, 1, 1],
                                  x0=N_batch1[:,0].tolist() + [35/100/1000])

start_index = 0
with Timer('Parameter estimation'):
    params = lmfit.Parameters()
    params.add('ks', value=1, min=0, max=150)
    params.add('kp', value=1, min=0, max=150)
    params.add('kg', value=1, min=0, max=150)
    params.add('s', value=1, min=0, max=20)
    params.add('p', value=1, min=0, max=20)
    params.add('g', value=1, min=0, max=20)
    params.add('gamma', value=1, min=0, max=20)
    params.add('kv', value=0.9, min=0.8, max=1)
    fit = lmfit.minimize(mechanistic_model.lmfit_loss, params=params,
                          args=(t_batch2[start_index:], N_batch2[:, start_index:],
                                N_batch2[:, start_index].tolist() + [35 / 100 / 1000], Tfunc_batch2),
                         method='leastsq')
    fit.params.pretty_print()

    # parameters = minimize(lambda parameters: mechanistic_model.loss(t_ref=t_batch2[start_index:],
    #                                                                 N_ref=N_batch2[:, start_index:],
    #                                                                 z=Tfunc_batch2,
    #                                                                 parameters=parameters,
    #                                                                 x0=N_batch2[:, start_index].tolist() + [35/100/1000]),
    #                       x0=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    #                       options={'disp': True})

start_index = 0
fig, ax = plt.subplots(figsize=(4*1.2, 3*1.2), dpi=400)
pred_train = mechanistic_model.solve_ode(t_steps=t_batch2,
                                         z=Tfunc_batch2,
                                         parameters=[fit.params['ks'].value,
                                                     fit.params['kp'].value,
                                                     fit.params['kg'].value,
                                                     fit.params['s'].value,
                                                     fit.params['p'].value,
                                                     fit.params['g'].value,
                                                     fit.params['gamma'].value,
                                                     fit.params['kv'].value],
                                         x0=N_batch2[:, 0].tolist() + [35 / 100 / 1000])
sns.lineplot(x=domain.axis[0].midpoints(), y=pred_train.y[:-1,-1]/np.sum(pred_train.y[:-1,-1]), label='Prediction, end-of-batch at t = '+str(np.round((t_batch2[0])/60/60, 1))+' hrs', ax=ax)
sns.lineplot(x=domain.axis[0].midpoints(), y=N_batch2[:,-1]/np.sum(N_batch2[:,-1]), label='Measured, end-of-batch at t = '+str(np.round(t_batch2[-1]/60/60,1))+' hrs', ax=ax)
plt.show()

start_index = 0
fig, ax = plt.subplots(figsize=(4*1.2, 3*1.2), dpi=400)
pred_val = mechanistic_model.solve_ode(t_steps=t_batch1[start_index:],
                                       z=Tfunc_batch1,
                                       parameters=[fit.params['ks'].value,
                                                     fit.params['kp'].value,
                                                     fit.params['kg'].value,
                                                     fit.params['s'].value,
                                                     fit.params['p'].value,
                                                     fit.params['g'].value,
                                                     fit.params['gamma'].value,
                                                     fit.params['kv'].value],
                                       x0=N_batch1[:, start_index].tolist() + [35 / 100 / 1000])
sns.lineplot(x=domain.axis[0].midpoints(), y=pred_val.y[:-1,-1]/np.sum(pred_val.y[:-1,-1]), label='Prediction at t = '+str(np.round((t_batch1[0])/60/60, 1))+' hrs', ax=ax)
sns.lineplot(x=domain.axis[0].midpoints(), y=N_batch1[:,-1]/np.sum(N_batch1[:,-1]), label='Measured at t = '+str(np.round(t_batch1[-1]/60/60,1))+' hrs', ax=ax)
plt.show()

error = np.sum(np.abs(pred_val.y[:-1,-1]/np.sum(pred_val.y[:-1,-1])-N_batch1[:,-1]/np.sum(N_batch1[:,-1])))/2*100
print(error)

pred_val.y[:30,-1]