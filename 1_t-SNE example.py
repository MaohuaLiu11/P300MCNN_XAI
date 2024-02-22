# 2024/02/21. Written by Maohua Liu
#In this demonstration, we created a P300 dataset, reduced its dimensionality using t-SNE, and ultimately visualized the dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define parameters
num_samples = 1000
num_time_points = 500
num_channels = 64
p300_peak_time = 300  # Time point where P300 peak occurs
p300_duration = 20  # Duration of P300 response
target_percentage = 0.1  # Target samples account for 10%

# Generate synthetic EEG data
eeg_data = np.random.randn(num_samples, num_time_points, num_channels)

# Generate labels
labels = np.zeros(num_samples, dtype=int)  # Initialize labels as non-target (0)
num_target_samples = int(num_samples * target_percentage)
target_indices = np.random.choice(num_samples, num_target_samples, replace=False)
labels[target_indices] = 1  # Set labels to 1 for target samples

# Add P300 response to target samples
for i in range(num_samples):
    if labels[i] == 1:
        # Add P300 response around p300_peak_time
        peak_start = max(p300_peak_time - p300_duration // 2, 0)
        peak_end = min(p300_peak_time + p300_duration // 2, num_time_points)
        eeg_data[i, peak_start:peak_end, :] += np.random.randn(peak_end - peak_start, num_channels)

# Display class distribution
print("Class Distribution:")
print("Non-target (0):", num_samples - num_target_samples)
print("Target (1):", num_target_samples)

# Assuming you have generated synthetic EEG data and labels as described in the previous code snippet

# Reshape the EEG data to have a shape of (num_samples, num_time_points * num_channels)
reshaped_data = np.reshape(eeg_data, (num_samples, -1))

# Perform t-SNE to reduce dimensionality to 2
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(reshaped_data)

# Visualize the t-SNE representation of the EEG data
plt.figure(figsize=(6, 4))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
plt.title("t-SNE Demonstration \n t-SNE Visualization of EEG Data")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label='Class')
plt.show()
