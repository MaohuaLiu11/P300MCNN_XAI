# 2024/02/22. Written by Maohua Liu
#This example showcases the implementation of a saliency map. 
#I generated synthetic EEG data for demonstration purposes.
#I constructed a basic neural network model to illustrate the concept.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to generate synthetic EEG-like data
def generate_eeg_data(num_samples, num_channels, num_time_steps):
    data = np.random.randn(num_samples, num_channels, num_time_steps)  # Gaussian noise
    labels = np.random.randint(0, 2, size=(num_samples,))
    return data, labels

# Function to compute the saliency map for a single input sample
def compute_saliency_map_single(model, input_data):
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_data, axis=0), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        loss = prediction[:, 0]  
    gradients = tape.gradient(loss, input_tensor)[0]  
    return gradients.numpy()


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

#compute the average saliency map
average_saliency_map = np.zeros((num_channels, num_time_steps))
average_input_sample = np.zeros((num_channels, num_time_steps))

for sample_index in range(len(X_test)):
    input_sample = X_test[sample_index]
    saliency_map = compute_saliency_map_single(model, input_sample)
    average_input_sample += input_sample#.numpy()
    average_saliency_map += saliency_map#.numpy()
    
average_input_sample /=len(X_test)
average_saliency_map /= len(X_test)


# Plot the original input data and its saliency map in matrix format
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(average_input_sample, aspect='auto', cmap='viridis')  # Transpose the data matrix
plt.title('Average EEG Signal')
plt.xlabel('Time Step')
plt.ylabel('Channel')
plt.yticks(np.arange(average_input_sample.shape[0]), np.arange(average_input_sample.shape[0]))  # Set y-axis ticks to channel numbers
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(average_saliency_map, aspect='auto', cmap='hot')  # Transpose the saliency map matrix
plt.title('Average Saliency Map')
plt.xlabel('Time Step')
plt.ylabel('Channel')
plt.yticks(np.arange(average_saliency_map.shape[0]), np.arange(average_saliency_map.shape[0]))  # Set y-axis ticks to channel numbers
plt.colorbar()

plt.tight_layout()
plt.show()
