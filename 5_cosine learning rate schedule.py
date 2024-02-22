# 2024/02/22. Written by Maohua Liu
#The cosine schedule is the same one used by Dataset D2 in the paper.
#The average of this cosine schedule is 0.017636

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_epochs = 10
learning_rate_max = 0.032
learning_rate_min = 0.00008

# Define cosine learning rate schedule function
def cosine_decay(epoch, lr_max, lr_min, num_epochs):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / num_epochs * np.pi))

# Generate learning rates for each epoch
learning_rates = [cosine_decay(epoch, learning_rate_max, learning_rate_min, num_epochs) for epoch in range(num_epochs)]

# Calculate the average learning rate
average_lr = np.mean(learning_rates)

# Plot the learning rate schedule
plt.plot(range(num_epochs), learning_rates)
plt.title('Cosine Learning Rate Schedule Demonstration\nAverage Learning Rate: {:.6f}'.format(average_lr))
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()
