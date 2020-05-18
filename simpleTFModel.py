import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def function(x1,x2):
    return -2*x1**3+2*x2**3+np.random.normal(loc=np.zeros_like(x1), scale=0.05)

x_1 = np.linspace(-0.5,0.5,20)
x_2 = np.linspace(-0.5,0.5,20)

x1, x2 = np.meshgrid(x_1, x_2)

y = function(x1, x2)

fig = plt.figure(dpi=600)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x1, x2, y, linewidth=0, antialiased=False)
fig.show()


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(200, activation="elu", name="Hidden_layer_1"),
        tf.keras.layers.Dense(100, activation="elu", name="Hidden_layer_2"),
        tf.keras.layers.Dense(1, name="Hidden_layer_3"),
    ]
)

model.compile(optimizer='Adam', loss='mean_squared_error')
x_train = np.concatenate([np.expand_dims(x1.flatten(),axis=-1),
                          np.expand_dims(x2.flatten(),axis=-1)],axis=1)
y_train = np.expand_dims(y.flatten(),axis=-1)

model.fit(x=x_train, y=y_train, epochs=10000, batch_size=50, verbose=1)


x_1 = np.linspace(-0.5,0.5,100)
x_2 = np.linspace(-0.5,0.5,100)

x_test1, x_test2 = np.meshgrid(x_1, x_2)

x_test = np.concatenate([np.expand_dims(x_test1.flatten(),axis=-1),
                          np.expand_dims(x_test2.flatten(),axis=-1)],axis=1)
y_test = model.predict(x=x_test).reshape([100,100])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_test1, x_test2, y_test, linewidth=0, antialiased=False)
fig.show()