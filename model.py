import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import queue

def create_cnn(input_shape, conv_layers, activation, filters, kernel_size, dense_units, dropout_rate):
    """
    Function to create a convolutional neural network.

    Parameters:
    - input_shape: Shape of input images (height, width, channels)
    - conv_layers: Number of convolutional layers
    - activation: Activation function to use
    - filters: Number of filters in convolutional layers
    - kernel_size: Size of convolutional kernels
    - dense_units: Number of units in dense layer
    - dropout_rate: Dropout rate for regularization

    Returns:
    - Compiled CNN model
    """
    model = models.Sequential()  # Creating sequential model
    for _ in range(conv_layers):
        # Adding convolutional layer with specified parameters
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding='same', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))  # Adding max pooling layer
    model.add(layers.Flatten())  # Flattening the output
    model.add(layers.Dense(dense_units, activation='relu'))  # Adding dense layer
    model.add(layers.Dropout(dropout_rate))  # Adding dropout layer for regularization
    model.add(layers.Dense(1, activation='sigmoid'))  # Adding output layer with sigmoid activation
    
    return model  # Returning the created model

def train_and_evaluate_model(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue):
    """
    Function to train and evaluate a CNN model.

    Parameters:
    - conv_layers: Number of convolutional layers
    - activation: Activation function to use
    - filters: Number of filters in convolutional layers
    - kernel_size: Size of convolutional kernels
    - dense_units: Number of units in dense layer
    - dropout_rate: Dropout rate for regularization
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - optimizer: Optimizer to use for training
    - loss: Loss function to use
    - progress_queue: Queue for reporting training progress

    Returns:
    - Test accuracy
    - Trained CNN model
    """
    train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1] for training data
    test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1] for test data

    # Generating training data from directory
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary',
        classes=['REAL', 'FAKE']
    )

    # Generating test data from directory
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary',
        classes=['REAL', 'FAKE']
    )

    input_shape = (32, 32, 3)  # Shape of input images
    model = create_cnn(input_shape, conv_layers, activation, filters, kernel_size, dense_units, dropout_rate)  # Creating CNN model

    model.compile(optimizer=optimizer,  # Compiling the model with specified optimizer and loss function
                  loss=loss,
                  metrics=['accuracy'])

    # Defining a custom callback to report training progress
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            progress_queue.put((batch, {'loss': logs['loss'], 'accuracy': logs['accuracy']}))

    # Training the model
    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=test_generator.samples // batch_size,
                        callbacks=[TrainingCallback()])

    # Evaluating the model on test data
    test_loss, test_acc = model.evaluate(test_generator)
    return test_acc, model  # Returning test accuracy and trained model
