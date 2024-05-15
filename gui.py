import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.filedialog
import threading
import queue
import numpy as np
from PIL import Image, ImageTk
from model import train_and_evaluate_model

epoch_counter = 0  # Counter to keep track of epochs
model = None  # Variable to store trained model

def on_train():
    try:
        global epoch_counter  # Using global variable for epoch counter
        epoch_counter = 0  # Resetting epoch counter
        conv_layers = int(conv_layers_var.get())  # Getting number of convolutional layers
        activation = activation_var.get()  # Getting activation function
        filters = int(filters_var.get())  # Getting number of filters
        kernel_size = int(kernel_size_var.get())  # Getting kernel size
        dense_units = int(dense_units_var.get())  # Getting number of neurons in dense layer
        dropout_rate = float(dropout_rate_var.get())  # Getting dropout rate
        epochs = int(epochs_var.get())  # Getting number of epochs
        batch_size = int(batch_size_var.get())  # Getting batch size
        optimizer = optimizer_var.get()  # Getting optimizer
        loss = loss_var.get()  # Getting loss function
        
        if conv_layers < 1:
            raise ValueError("Number of convolutional layers must be greater than 0")
        
        progress_bar["maximum"] = 100000 // batch_size  # Setting maximum value for progress bar
        progress_queue = queue.Queue()  # Creating a queue for progress updates
        threading.Thread(target=train_model, args=(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue)).start()  # Starting training in a separate thread
        root.after(100, check_progress, progress_queue)  # Scheduling progress check
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_classify():
    global model
    try:
        image_path = image_path_var.get()  # Getting image path
        if not image_path:
            raise ValueError("Please provide a path to the image.")
        
        if model is None:
            raise ValueError("Model is not trained yet.")
        
        img = Image.open(image_path)  # Opening image
        img = img.resize((32, 32))  # Resizing image
        img = img.convert('RGB')  # Converting to RGB
        img = np.array(img) / 255.0  # Normalizing image
        img = np.expand_dims(img, axis=0)  # Adding batch dimension

        prediction = model.predict(img)  # Making prediction
        result_label.config(text="Real" if prediction[0][0] < 0.5 else "Fake")  # Displaying result
        prediction_label.config(text=f'Predicted value {prediction[0][0]:.4f}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

def train_model(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue):
    try:
        global model
        test_acc, model = train_and_evaluate_model(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue)  # Training the model
        messagebox.showinfo("Result", f'Test accuracy: {test_acc}')  # Displaying test accuracy
        test_acc_label.config(text=f'Final model accuracy {test_acc:.4f}') # Displaying test accuracy
    except Exception as e:
        messagebox.showerror("Error", str(e))

def check_progress(progress_queue):
    global epoch_counter
    try:
        while True:
            iteration, logs = progress_queue.get_nowait()  # Getting progress update
            update_progress(iteration, logs)  # Updating progress
            if iteration % progress_bar['maximum'] == 0 and epoch_counter < int(epochs_var.get()):
                epoch_counter += 1  # Updating epoch counter
    except queue.Empty:
        pass
    root.after(100, check_progress, progress_queue)  # Scheduling next progress check

def update_progress(iteration, logs):
    global epoch_counter
    progress_bar["value"] = iteration  # Updating progress bar value
    progress_label.config(text=f"Epoch: {epoch_counter}/{epochs_var.get()}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")  # Updating progress label

def browse_image():
    filepath = tk.filedialog.askopenfilename()  # Opening file dialog for image selection
    if filepath:
        image_path_var.set(filepath)  # Setting image path variable
        display_image(filepath)  # Displaying selected image

def display_image(filepath):
    img = Image.open(filepath)  # Opening image
    img = img.resize((200, 200))  # Resizing image
    img = ImageTk.PhotoImage(img)  # Converting image to Tkinter format
    image_label.img = img   # Keeping reference to image to prevent garbage collection
    image_label.configure(image=img)  # Displaying image

# Creating GUI
root = tk.Tk()  # Creating main window
root.title("CNN GUI")  # Setting title

# Configure grid for the main window
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Divide main frame into two sections
left_frame = ttk.Frame(main_frame, padding="5")
left_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S))

right_frame = ttk.Frame(main_frame, padding="5")
right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

# Neural network parameters
params = [
    ("Number of convolutional layers:", "2", tk.StringVar()),
    ("Activation function:", "relu", tk.StringVar(), ["relu", "sigmoid", "tanh", "softmax"]),
    ("Number of filters:", "32", tk.StringVar()),
    ("Kernel size:", "3", tk.StringVar()),
    ("Number of neurons in dense layer:", "512", tk.StringVar()),
    ("Dropout rate:", "0.5", tk.StringVar()),
    ("Number of epochs:", "10", tk.StringVar()),
    ("Batch size:", "32", tk.StringVar()),
    ("Optimizer:", "adam", tk.StringVar(), ["adam", "sgd", "rmsprop"]),
    ("Loss function:", "binary_crossentropy", tk.StringVar(), ["binary_crossentropy", "mean_squared_error"])
]

row = 0
vars_dict = {}

for param in params:
    label = ttk.Label(left_frame, text=param[0])  # Creating label for parameter
    label.grid(row=row, column=0, sticky=tk.W, pady=5)  # Placing label in left frame

    if len(param) == 4:  # If parameter has options
        var = param[2]
        var.set(param[1])  # Setting default value
        vars_dict[param[0]] = var  # Storing variable in dictionary
        option_menu = ttk.OptionMenu(left_frame, var, param[1], *param[3])  # Creating option menu
        option_menu.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)  # Placing option menu in left frame
    else:
        var = param[2]
        var.set(param[1])  # Setting default value
        vars_dict[param[0]] = var  # Storing variable in dictionary
        entry = ttk.Entry(left_frame, textvariable=var)  # Creating entry for input
        entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)  # Placing entry in left frame
    
    row += 1  # Moving to next row

conv_layers_var = vars_dict["Number of convolutional layers:"]
activation_var = vars_dict["Activation function:"]
filters_var = vars_dict["Number of filters:"]
kernel_size_var = vars_dict["Kernel size:"]
dense_units_var = vars_dict["Number of neurons in dense layer:"]
dropout_rate_var = vars_dict["Dropout rate:"]
epochs_var = vars_dict["Number of epochs:"]
batch_size_var = vars_dict["Batch size:"]
optimizer_var = vars_dict["Optimizer:"]
loss_var = vars_dict["Loss function:"]

train_button = ttk.Button(left_frame, text="Train model", command=on_train)  # Creating button to train model
train_button.grid(row=row, column=0, columnspan=2, pady=20)  # Placing button in left frame

# Progress bar and label
row += 1
progress_bar = ttk.Progressbar(left_frame, orient="horizontal", length=400, mode="determinate")  # Creating progress bar
progress_bar.grid(row=row, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))  # Placing progress bar in left frame

progress_label = ttk.Label(left_frame, text="")  # Creating label for progress
progress_label.grid(row=row+1, column=0, columnspan=2, pady=5)  # Placing label in left frame

# Final model accuracy
row += 2
test_acc_label = ttk.Label(left_frame, text="")  # Creating label for final model accuracy
test_acc_label.grid(row=row, column=0, columnspan=2, pady=5)  # Placing label in left frame

# Classify image
row = 0
classify_title_label = ttk.Label(right_frame, text="Image Classification")  # Creating label for image classification
classify_title_label.grid(row=row, column=0, columnspan=2, pady=10)  # Placing label in right frame

# Input image path
row += 1
image_path_var = tk.StringVar()  # Creating variable for image path
image_entry = ttk.Entry(right_frame, textvariable=image_path_var)  # Creating entry for image path
image_entry.grid(row=row, column=0, columnspan=2, pady=5, padx=5, sticky=(tk.W, tk.E))  # Placing entry in right frame

# Browse button for searching images in local OS
browse_button = ttk.Button(right_frame, text="Browse", command=browse_image)  # Creating browse button
browse_button.grid(row=row+1, column=0, columnspan=2, pady=5, padx=5, sticky=(tk.W, tk.E))  # Placing browse button in right frame

# Button which executes classification
classify_button = ttk.Button(right_frame, text="Classify Image", command=on_classify)  # Creating classify button
classify_button.grid(row=row+2, column=0, columnspan=2, pady=5)  # Placing classify button in right frame

# Display input image
row += 3
image_label = ttk.Label(right_frame)  # Creating label for image
image_label.grid(row=row, column=0, columnspan=2, padx=5, pady=5)  # Placing label in right frame

# Result label
result_label = ttk.Label(right_frame, text="")  # Creating label for result
result_label.grid(row=row+1, column=0, columnspan=2, pady=5)  # Placing label in right frame

# Prediction value label
row += 2
prediction_label = ttk.Label(right_frame, text="")  # Creating label for prediction value
prediction_label.grid(row=row, column=0, columnspan=2, pady=5)  # Placing label in right frame


# Configure grid for elements in the left and right frames
for frame in [left_frame, right_frame]:
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    for i in range(len(params) + 3):
        frame.rowconfigure(i, weight=1)

root.mainloop()  # Running the main event loop

