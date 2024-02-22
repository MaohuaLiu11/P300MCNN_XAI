# 2024/02/21. Written by Maohua Liu
#This serves as an illustration of utilizing PCA for reducing the dimensionality of weights. 
#Here, we construct a random neural network and employ PCA to visualize the weights of its initial layer.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from sklearn.decomposition import PCA

# Generate some dummy data for training
np.random.seed(0)
X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features
y_train = np.random.randint(0, 2, size=(1000,))  # Binary classification labels

# Define a custom callback to store and visualize dimension-reduced weights after each epoch
class VisualizeWeightsCallback(Callback):
    def __init__(self):
        self.reduced_weights_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        # Get the weights of the first layer and flatten them
        first_layer_weights = self.model.layers[0].get_weights()[0]
        
        # Perform PCA to reduce the dimensionality of the weights to 2
        pca = PCA(n_components=2)
        reduced_weights = pca.fit_transform(first_layer_weights.T)
        
        # Store the reduced weights
        self.reduced_weights_per_epoch.append(reduced_weights)

    def plot_dimension_reduced_weights(self):
        # Visualize all stored dimension-reduced weights together
        plt.figure(figsize=(8, 6))
        for epoch, reduced_weights in enumerate(self.reduced_weights_per_epoch):
            plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], label=f'Epoch {epoch+1}')
        plt.title("PCA Demonstration: may need to run it more than once \n Visualization of First Layer Weights \n There are 4 neurons in the first layer")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()

# Define the model
model = Sequential([
    Dense(4, activation='relu', input_shape=(10,)),
    Dense(2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create an instance of the custom callback
visualize_weights_callback = VisualizeWeightsCallback()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[visualize_weights_callback])

# After training, visualize all stored dimension-reduced weights together
visualize_weights_callback.plot_dimension_reduced_weights()
