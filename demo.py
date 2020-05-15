## Loading modules
from data.Data import Data
from domain.Domain import Domain
from hybrid_model.System import System, OdeSettings, LossSettings, RateSettings
from hybrid_model.HybridModel import HybridModel
from hybrid_model.TimeSeriesPair import TimeSeriesPair
from hybrid_model.Illustration import Illustration
import tensorflow as tf

# Construct discretized domain object for hybrid model
domain = Domain(name='Domain')
domain.add_axis(x_min=5, x_max=150, m=30, disc_by='FeretMean', name='FeretMean')

# Set ode and loss settings
ode_settings = OdeSettings(variable_stepsize=True, time_steps=2, rel_tol=1e-6, abs_tol=1e-6)
loss_settings = LossSettings(geometry='Sphere', loss_type='Number')
rate_settings = RateSettings(layer_activations=['elu', 'elu', 'linear'], layer_neurons=[33, 32, 31],
                             scaling_factors={'nucleation': 1/1000, 'growth': 1/1000, 'shrinkage': 1/10000,
                                              'agglomeration': 1/1000, 'breakage': 1/1000})

# Define model system
system = System(case="Laboratory lactose case study", domain=domain, ode_settings=ode_settings,
                loss_settings=loss_settings, rate_settings=rate_settings, dilution=False,
                regularization=1, normalize=True)

# Adding sensors
system.add_sensor(name='Temperature', measured=True, controlled=True, unit='C')
system.add_sensor(name='Concentration', measured=True, controlled=False, unit='g/µL')

# Activate phenomena
system.activate_phenomena(['nucleation', 'growth'])

# Create data-set and set up data-shuffler
data = Data(case_id='Demo data')
data.load_from_pickle('demo_data')
time_series_pair = TimeSeriesPair(data=data, system=system)

# Split training and validation data
data.set_batch_pool(pool_batch_id=['Demo batch 0', 'Demo batch 1', 'Demo batch 2', 'Demo batch 3'], pool_type='Training')
data.set_batch_pool(pool_batch_id=['Demo batch 4', 'Demo batch 5', 'Demo batch 6', 'Demo batch 7', 'Demo batch 8', 'Demo batch 9'], pool_type='Validation')
data.set_batch_pool(pool_batch_id=['Demo batch 4'], pool_type='Test')

# Set up hybrid training model
hybrid_model = HybridModel(system=system)

# Compile hybrid model
hybrid_model.training_model.compile(loss=hybrid_model.loss_model.loss, optimizer='Adam')

# Generate shuffled training and evaluation data
training_data = time_series_pair.shuffle(pool_type=['Training'], delta_t_critical=20*60)
validation_data = time_series_pair.shuffle(pool_type=['Validation'], delta_t_critical=20*60)
test_data = time_series_pair.shuffle(pool_type=['Test'], min_step=1, max_step=1)

# Create training data
[x_train, y_train] = hybrid_model.model_data(training_data)
[x_val, y_val] = hybrid_model.model_data(validation_data)

# Get reference losses
hybrid_model.calculate_reference_loss(x_train, y_train, category='train')
hybrid_model.calculate_reference_loss(x_val, y_val, category='val')

# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

hybrid_model.training_history = hybrid_model.training_model.fit(x=x_train, y=y_train,
                                                                validation_data=(x_val, y_val),
                                                                callbacks=[es],
                                                                epochs=200, batch_size=50, verbose=1).history

# Illustrating the results
illustrator = Illustration(domain)
fig_a = illustrator.plot_explained_variation(history=hybrid_model.training_history,
                                             ref_loss_train=hybrid_model.ref_loss_train,
                                             ref_loss_val=hybrid_model.ref_loss_val)

fig_b = illustrator.plot_multi_prediction(hybrid_model=hybrid_model, prediction_data=test_data,
                                          start_step=[0, 10, 20], target_step=30, relative=True)
