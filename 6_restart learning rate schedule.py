# 2024/02/22. Written by Maohua Liu
#This restart schedule is the same one used by Dataset D4 for cross-subject P300 detection in the paper.
#The average of this restart schedule is 0.026526

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class RestartScheduler(Callback):
    '''Restart learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = RestartScheduler(min_lr=5e-5,
                                        max_lr=5e-2,
                                        steps_per_epoch=np.ceil(epoch_size/batch_size),
                                        lr_decay=0.9,
                                        cycle_length=1,
                                        mult_factor=2)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.trn_iterations = 0.
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {'lr': []}  # Initialize history with 'lr' key

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the maximum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.trn_iterations += 1

        self.history['lr'].append(K.get_value(self.model.optimizer.lr))  # Append lr to history

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1

        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay

    def on_train_end(self, logs={}):
        '''No action needed at the end of training.'''
        pass


# Generate synthetic EEG-like data
def generate_eeg_data(num_samples, num_channels, num_time_steps):
    data = np.random.randn(num_samples, num_channels, num_time_steps)  # Gaussian noise
    labels = np.random.randint(0, 2, size=(num_samples,))
    return data, labels

# Define model architecture
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Generate synthetic EEG data
num_samples = 1000
num_channels = 3
num_time_steps = 128
X, y = generate_eeg_data(num_samples, num_channels, num_time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for the scheduler
batch_size = 32
epoch_size = len(X_train)
steps_per_epoch = np.ceil(epoch_size / batch_size)
min_lr = 5e-5
max_lr = 5e-2
lr_decay = 0.9
cycle_length = 1
mult_factor = 2

# Create and compile the model
model = create_model((num_channels, num_time_steps))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create the restart learning rate scheduler
schedule = RestartScheduler(min_lr=min_lr,
                            max_lr=max_lr,
                            steps_per_epoch=steps_per_epoch,
                            lr_decay=lr_decay,
                            cycle_length=cycle_length,
                            mult_factor=mult_factor)

# Train the model
history = model.fit(X_train, y_train, epochs=7, batch_size=batch_size, validation_split=0.2, callbacks=[schedule])

average_lr = np.mean(schedule.history['lr'])
# Plot the learning rate schedule
plt.figure(figsize=(8, 6))  # Set the size of the figure
plt.plot(np.arange(len(schedule.history['lr'])), schedule.history['lr'])
plt.title('Restart Learning Rate Schedule Demonstration\nmin_lr=5e-5, max_lr=5e-2, mult_factor=2, lr_decay=0.9\nAverage Learning Rate: {:.6f}'.format(average_lr))
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.xticks(np.arange(0, len(schedule.history['lr']), len(schedule.history['lr']) // 7), np.arange(1, 8))
plt.show()
