from keras.models import load_model
import os

# Print current working directory
print("Current Directory:", os.getcwd())

# Load model
model_path = r'E:/SRM/minor project/project/Human_Scream_Detection/model.keras'
print("Loading model from:", model_path)

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except ValueError as e:
    print("Error loading model:", e)
