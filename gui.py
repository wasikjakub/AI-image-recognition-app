import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.filedialog
import threading
import queue
import numpy as np
from PIL import Image, ImageTk
import requests
import io
from model import train_and_evaluate_model

epoch_counter = 0  # Counter to keep track of epochs
model = None  # Variable to store trained model

def on_train():
    try:
        global epoch_counter  # Using global variable for epoch counter
        epoch_counter = 0  # Resetting epoch counter
        conv_layers = int(conv_layers_var.get())  # Getting number of convolutional layers
        activation = activation_var.get()  # Getting activation function
        filters = [int(filters_vars[i].get()) for i in range(conv_layers)]  # Getting number of filters for each layer
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
        result_label.config(text="REAL" if prediction[0][0] < 0.5 else "FAKE")  # Displaying result
        prediction_label.config(text=f'Predicted value: {prediction[0][0]:.4f}')
    except Exception as e:
        messagebox.showerror("Error", str(e))


def train_model(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue):
    try:
        global model
        test_acc, model = train_and_evaluate_model(conv_layers, activation, filters, kernel_size, dense_units, dropout_rate, epochs, batch_size, optimizer, loss, progress_queue)  # Training the model
        messagebox.showinfo("Result", f'Test accuracy: {test_acc}')  # Displaying test accuracy
        test_acc_label.config(text=f'Final model accuracy: {test_acc:.4f}') # Displaying test accuracy
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


def load_image_from_url(url):
    response = requests.get(url, stream=True)  # Getting image from URL
    response.raise_for_status()  # Checking for any errors in response
    image_data = response.content  # Getting image content

    # Opening image using PIL
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((200, 200))  # Resizing image to fit in GUI

    # Converting image to Tkinter format
    img_tk = ImageTk.PhotoImage(img)

    return img_tk


def on_load_from_url():
    try:
        url = url_entry.get()  # Getting URL from entry
        if not url:
            raise ValueError("Please provide a URL.")

        img_tk = load_image_from_url(url)  # Loading image from URL
        image_label.img = img_tk  # Keeping reference to image to prevent garbage collection
        image_label.configure(image=img_tk)  # Displaying image

        # Clearing previous result
        result_label.config(text="")
        prediction_label.config(text="")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def display_image(filepath):
    img = Image.open(filepath)  # Opening image
    img = img.resize((200, 200))  # Resizing image
    img = ImageTk.PhotoImage(img)  # Converting image to Tkinter format
    image_label.img = img   # Keeping reference to image to prevent garbage collection
    image_label.configure(image=img)  # Displaying image


def update_filters_entries(*args):
    # Clear existing filter entries
    for widget in filters_frame.winfo_children():
        widget.destroy()

    # Create new filter entries based on the number of convolutional layers
    try:
        num_layers = int(conv_layers_var.get())
        filters_vars.clear()
        for i in range(num_layers):
            var = tk.StringVar()
            var.set("32")  # Default filter value
            filters_vars.append(var)
            label = ttk.Label(filters_frame, text=f"Filters for layer {i+1}:")
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(filters_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2)
    except ValueError:
        pass  # Invalid input for number of layers


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
kernel_size_var = vars_dict["Kernel size:"]
dense_units_var = vars_dict["Number of neurons in dense layer:"]
dropout_rate_var = vars_dict["Dropout rate:"]
epochs_var = vars_dict["Number of epochs:"]
batch_size_var = vars_dict["Batch size:"]
optimizer_var = vars_dict["Optimizer:"]
loss_var = vars_dict["Loss function:"]

# Filters input frame
filters_frame = ttk.Frame(left_frame, padding="5")
filters_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
filters_vars = []

# Update filters entries when the number of convolutional layers changes
conv_layers_var.trace_add("write", update_filters_entries)
update_filters_entries()

# Buttons
train_button = ttk.Button(left_frame, text="Train", command=on_train)
train_button.grid(row=row+1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Progress bar
progress_bar = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress_bar.grid(row=row+2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Progress label
progress_label = ttk.Label(left_frame, text="")
progress_label.grid(row=row+3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

# Test accuracy label
test_acc_label = ttk.Label(left_frame, text="")
test_acc_label.grid(row=row+4, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))

row = 0
# Image path from PC
image_path_var = tk.StringVar()
image_path_entry = ttk.Entry(right_frame, textvariable=image_path_var)
image_path_entry.grid(row=row, column=0, columnspan=2, pady=5, padx=5, sticky=(tk.W, tk.E))
browse_button = ttk.Button(right_frame, text="Browse", command=browse_image)
browse_button.grid(row=row+1, column=0, columnspan=2, pady=5)

# Input URL
row += 1
url_entry_var = tk.StringVar()
url_entry = ttk.Entry(right_frame, textvariable=url_entry_var)
url_entry.grid(row=row+2, column=0, columnspan=2, pady=5, padx=5, sticky=(tk.W, tk.E))

# Button to load image from URL
row += 1
load_button = ttk.Button(right_frame, text="Load from URL", command=on_load_from_url)
load_button.grid(row=row+3, column=0, columnspan=2, pady=5)

# Right frame for displaying images and results
image_label = ttk.Label(right_frame, text="")
image_label.grid(row=row+4, column=1, padx=10, pady=10, sticky="nsew")

# Classify button
classify_button = ttk.Button(right_frame, text="Classify", command=on_classify)
classify_button.grid(row=row+5, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))


# Result label
result_label = ttk.Label(right_frame, text="")
result_label.grid(row=row+6, column=0, padx=10, pady=10)

# Prediction label
prediction_label = ttk.Label(right_frame, text="")
prediction_label.grid(row=row+7, column=0, padx=10, pady=10)

root.mainloop()
