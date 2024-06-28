# AI image recognition app

This project provides a graphical user interface (GUI) for training a Convolutional Neural Network (CNN) model and classifying images as AI-generated or real. The application uses `tkinter` for the GUI, `PIL` for image processing, and `requests` for fetching images from URLs.

## Table of Contents
  
  - [Installation](#installation)
  - [Usage](#usage)
  - [About](#about)
  - [Features](#features)
  - [GUI Components](#gui-components)
  - [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AiR-ISZ-Gr1/deep_learning_CNN.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download dataset from kaggle:

   ```bash
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data
   ```

4. Unzip dataset to `./data` folder


## Usage

Run the main script to start the GUI application:
```sh
python gui.py
```

## About

As AI-generated images become increasingly realistic, distinguishing between authentic and fabricated visuals has become a critical challenge. This project aims to address this issue by providing an intuitive and accessible tool for detecting AI-generated images. By leveraging a Convolutional Neural Network (CNN) within a user-friendly GUI, the application empowers users to train and customize their own image classification models. 

This enables individuals and organizations to enhance their capabilities in identifying fake images, contributing to the broader effort of maintaining truth and authenticity in the digital age. Through continuous improvement and adaptation of AI detection technologies, we can better safeguard against the spread of misinformation and uphold the integrity of visual media.


## Features

- **Train CNN Model**: Configure and train a CNN model with customizable parameters.
- **Image Classification**: Classify images as "REAL" or "FAKE" using a trained model.
- **Load Images**: Load images from a local file system or a URL.


## GUI Components

### Model Training Section

1. **Number of convolutional layers**: Set the number of convolutional layers.
2. **Activation function**: Choose the activation function (e.g., relu, sigmoid).
3. **Kernel size**: Set the kernel size for convolutional layers.
4. **Number of neurons in dense layer**: Set the number of neurons in the dense layer.
5. **Dropout rate**: Set the dropout rate to prevent overfitting.
6. **Number of epochs**: Set the number of epochs for training.
7. **Batch size**: Set the batch size for training.
8. **Optimizer**: Choose the optimizer (e.g., adam, sgd).
9. **Loss function**: Choose the loss function (e.g., binary_crossentropy, mean_squared_error).
10. **Train button**: Start training the model.

### Image Classifier Section

1. **Image path (local)**: Enter or browse the path of the image file from the local file system.
2. **URL**: Enter the URL of the image.
3. **Load from URL**: Load the image from the entered URL.
4. **Classify Local**: Classify the locally loaded image.
5. **Classify URL**: Classify the image loaded from the URL.
6. **Result display**: Display the classification result ("REAL" or "FAKE").
7. **Predicted value display**: Display the predicted value.

## License

This project is licensed under the MIT License.
