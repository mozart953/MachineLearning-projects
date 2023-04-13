import matplotlib.pyplot as plt
import readFile as rF

import tensorflow as tf

def plot_loss(history):
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(x_train, y_train, num_modes, dropout_prob, lr, batch_size, epochs):

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_modes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_modes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                     metrics=['accuracy'])
    history = nn_model.fit(
        rF.x_train, rF.y_train,
        epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
    )
    return nn_model, history


# history = nn_model.fit(
#     rF.x_train, rF.y_train,
#     epochs=100, batch_size=32, validation_split=0.2, verbose=0
# )
#
# plot_loss(history)
# plot_accuracy(history)

