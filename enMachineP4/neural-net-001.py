import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Datos de entrenamiento
celsius = np.array([-40, 0, 10, 37, 100]).reshape(-1, 1)
fahrenheit = np.array([-40, 32, 50, 98.6, 212]).reshape(-1, 1)

# Normalización de datos
scaler_celsius = MinMaxScaler()
scaler_fahrenheit = MinMaxScaler()

celsius_normalized = scaler_celsius.fit_transform(celsius)
fahrenheit_normalized = scaler_fahrenheit.fit_transform(fahrenheit)

# Parámetros de la red
input_size = 1
hidden_size = 5
output_size = 1
learning_rate = 0.01

# Inicializar pesos y sesgos
weights1 = np.random.randn(input_size, hidden_size) * 0.01
biases1 = np.zeros((1, hidden_size))
weights2 = np.random.randn(hidden_size, output_size) * 0.01
biases2 = np.zeros((1, output_size))

# Entrenamiento
for epoch in range(10000):
    # Forward propagation
    hidden_layer = np.dot(celsius_normalized, weights1) + biases1
    hidden_layer = np.maximum(0, hidden_layer)  # Función de activación ReLU
    output = np.dot(hidden_layer, weights2) + biases2

    # Backpropagation
    error = fahrenheit_normalized - output
    d_output = -error
    d_hidden = np.dot(d_output, weights2.T) * (hidden_layer > 0)

    # Verificación de valores
    if np.any(np.isnan(d_hidden)) or np.any(np.isinf(d_hidden)):
        print("d_hidden contiene NaNs o Infs")
        break

    # Actualizar pesos y sesgos
    weights2 -= learning_rate * np.dot(hidden_layer.T, d_output)
    biases2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
    weights1 -= learning_rate * np.dot(celsius_normalized.T, d_hidden)
    biases1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # Opcional: imprimir cada 100 épocas para seguimiento
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Error = {np.mean(np.abs(error))}")

# Predecir una nueva temperatura
new_celsius = np.array([[25]])
new_celsius_normalized = scaler_celsius.transform(new_celsius)
hidden_layer = np.dot(new_celsius_normalized, weights1) + biases1
hidden_layer = np.maximum(0, hidden_layer)
predicted_fahrenheit_normalized = np.dot(hidden_layer, weights2) + biases2

# Desnormalizar la predicción
predicted_fahrenheit = scaler_fahrenheit.inverse_transform(predicted_fahrenheit_normalized)
print(predicted_fahrenheit)
