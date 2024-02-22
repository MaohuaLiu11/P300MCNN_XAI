# 2024/02/22. Written by Maohua Liu
#This example showcases the implementation of average activation map. 
#I generated synthetic EEG data for demonstration purposes.
#I constructed a basic neural network model to illustrate the concept.
# The result looks very simple because the first layer of the model only has 64 neurons.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to generate synthetic EEG-like data
def generate_eeg_data(num_samples, num_channels, num_time_steps):
    data = np.random.randn(num_samples, num_channels, num_time_steps)  # Gaussian noise
    labels = np.random.randint(0, 2, size=(num_samples,))
    return data, labels

# Generate synthetic EEG data
num_samples = 1000
num_channels = 3  # Example: Fp1, Fp2, Fz
num_time_steps = 128  # Number of time steps per channel
X, y = generate_eeg_data(num_samples, num_channels, num_time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network classifier
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_channels, num_time_steps)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Extract activation maps of the first layer
activation_model = tf.keras.Model(inputs=model.input, outputs=model.layers[1].output)
average_activation_map = np.zeros((64,))

for sample_index in range(len(X_test)):
    input_sample = X_test[sample_index]
    activation_map = activation_model.predict(np.expand_dims(input_sample, axis=0))[0]
    average_activation_map += activation_map

average_activation_map /= len(X_test)

# Plot the average activation map
plt.figure(figsize=(6, 4))
plt.plot(average_activation_map)
plt.title('Average Activation Map Demonstration \n Average Activation Map of the First Layer')
plt.xlabel('Neuron')
plt.ylabel('Activation')
plt.show()


