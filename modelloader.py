from tensorflow import keras
from keras.models import load_model
import pandas as pd
from scipy.io.wavfile import read
import os

# Load your pre-trained model
model_path = r'E:/SRM/minor project/project/Human_Scream_Detection/model.keras'

# Ensure the model is saved before trying to load it
if not os.path.exists(model_path):
    model = keras.Sequential()  # Replace this with your model definition if needed
    # Model architecture definition goes here
    model.save(model_path)  # Save the model if it doesn't exist

model = load_model(model_path)  # Load the existing model

def process_file(filename):
    arr = []
    print(f"Processing file: {filename}")
    
    try:
        data, rs = read(filename)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # Read the suitable length for the model
    try:
        with open("input dimension for model.txt", "r") as file:
            suitable_length_for_model = int(file.read().strip())
    except Exception as e:
        print(f"Error reading input dimension file: {e}")
        return False

    rs = rs.astype(float)

    # Ensure we are slicing the array correctly
    if len(rs) < suitable_length_for_model + 1:
        print(f"Error: The audio signal is shorter than the expected length of {suitable_length_for_model + 1}.")
        return False

    rs = rs[0:suitable_length_for_model + 1]
    a = pd.Series(rs)
    arr.append(a)
    df = pd.DataFrame(arr)

    # Prepare input for the model
    X2 = df.iloc[0:, 1:]  # Adjust if necessary
    predictions = model.predict(X2)
    rounded = [round(x[0]) for x in predictions]

    print("Predicted value is:", rounded)
    return rounded == [1.0]

# Example usage
result = process_file("/home/themockingjester/PycharmProjects/multilayer_perceptron_modal_for_project_Human_Screem_Detection/venv/positive/damm_6.wav")
if result:
    print("Scream detected.")
else:
    print("No scream detected.")
