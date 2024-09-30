import numpy as np
import librosa
import pickle
import sounddevice as sd
import wavio
import os

def record_audio(filename, duration):
    """
    Record audio from the microphone and save it to a WAV file.
    
    Args:
    - filename (str): Path to save the recorded audio.
    - duration (int): Duration of the recording in seconds.
    """
    if not filename:
        print("Error: Filename cannot be empty.")
        return  # Exit if filename is empty

    print("Recording...")
    # Record audio
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save the recorded audio
    try:
        wavio.write(filename, recording, 44100, sampwidth=2)
        print(f"Audio saved as {filename}")
    except Exception as e:
        print(f"Error saving audio file: {e}")

def testing_unit(filename):
    """
    Load a WAV file and extract its MFCC features.
    
    Args:
    - filename (str): Path to the audio file.
    
    Returns:
    - np.ndarray: Mean MFCC features extracted from the audio.
    """
    # Load the audio file
    signal, sr = librosa.load(filename, sr=None)  # Set sr=None to preserve original sample rate
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs.reshape(1, -1)  # Reshape to ensure it has the right dimensions for prediction

def svm_process(filename):
    """
    Process the audio file with SVM models to classify the sound.
    
    Args:
    - filename (str): Path to the audio file.
    
    Returns:
    - bool: True if detected as a scream, False for speech, or "Noise" for non-human sounds.
    """
    # Record audio
    record_audio(filename, 5)  # Record for 5 seconds (you can change the duration)

    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return "File not found"

    # Load phase 1 model (Noise vs. Speech)
    with open('svm_based_model/phase1_model.sav', 'rb') as model_file:
        phase1_model = pickle.load(model_file)
    
    # Predict using phase 1 model
    phase1_result = phase1_model.predict(testing_unit(filename))
    
    if phase1_result[0] == 1:  # If result is speech (1)
        print("Phase-1 clear: Speech detected.")
        
        # Load phase 2 model (Scream vs. Speech)
        with open('svm_based_model/phase2_model.sav', 'rb') as model_file:
            phase2_model = pickle.load(model_file)
        
        # Predict using phase 2 model
        phase2_result = phase2_model.predict(testing_unit(filename))
        
        if phase2_result[0] == 1:  # If result is scream (1)
            print("Phase-2 clear: Scream detected.")
            return True  # Detected scream
        else:
            print("Detected speech, not a scream.")
            return False  # Detected speech, not a scream
    else:
        print("Detected noise.")
        return "Noise"  # Detected noise

# Example usage:
if __name__ == "__main__":
    filename = "output.wav"  # Specify your desired output filename
    result = svm_process(filename)
    print(f"Detection Result: {result}")
