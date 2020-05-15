from reference_model.Lactose.LactoseCrystallizer import LactoseCrystallizer
from data.Data import Data, Batch
from domain.Domain import Domain
import numpy as np
import seaborn as sns
from data.Illustration import Illustration
import matplotlib.pyplot as plt
sns.set()

# Construct discretized domain object for hybrid model
domain = Domain(name='Domain')
domain.add_axis(x_min=5, x_max=150, m=300, disc_by='FeretMean', name='FeretMean')

# Construct artificial crystallizer
crystallizer = LactoseCrystallizer(domain=domain, save_intermediates=False, verbose=False)

# Accumulated data
data = Data(case_id='Demo data')
for batch_id in range(10):
    data.add_batch(Batch(batch_id='Demo batch '+str(batch_id)))
    initial_measurement = crystallizer.start_new_batch(N0=np.concatenate(([100], np.zeros(domain.axis[0].m-1))),
                                                       C0=0.000445085 + np.random.normal(loc=0, scale=0.00005),
                                                       T0=50 + np.random.normal(loc=0, scale=5),
                                                       step_size=5*60, noise_level=0)
    data.batches[-1].add_measurement(initial_measurement)
    for _ in range(30):
        data.batches[-1].add_measurement(crystallizer.get_next_measurement(-0.00444))
data.save_to_pickle('demo_data')

illustrator = Illustration(data=data)
illustrator.plot_count_time(variable_name='FeretMean', category='particle_analysis')


# fig, ax = plt.subplots()
# sns.distplot(data.batches[-1].measurements[-1].particle_analysis_sensors[0].value, bins=domain.axis[0].edges(), ax=ax)
# plt.show()
# print(np.median(data.batches[-1].measurements[-1].particle_analysis_sensors[0].value))