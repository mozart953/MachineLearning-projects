import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


celcius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype= float)

# capa = tf.keras.layers.Dense(units=1,input_shape=[1])
# modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3 ,salida])



modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("training...")

historial = modelo.fit(celcius, fahrenheit, epochs=1000, verbose=False)

print('trained')

plt.xlabel("# Epoca")
plt.ylabel("Losted magnitude")
plt.plot(historial.history["loss"])

print("prediction")
resultado = modelo.predict([100.0])
print("The result is " + str(resultado) + "Fahrenheit")

print("model's internal variables")

#print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(oculta3.get_weights())
print(salida.get_weights())