import pyaudio
import numpy as np
import librosa
from sklearn.svm import SVC
import time

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 70

# Initialize PyAudio
p = pyaudio.PyAudio()

# Predefined classes for noise type
NOISE_CLASSES = ['background', 'speech', 'mechanical']

# Sample SVM model for demonstration (in a real case, train this with a dataset)
def train_svm_model():
    X = np.array([
        [0.5, 1.2],  # Background noise features (example)
        [2.1, 3.5],  # Speech features (example)
        [4.3, 5.1]   # Mechanical noise features (example)
    ])
    y = np.array([0, 1, 2])  # Corresponding labels for the noise classes
    
    # Train an SVM model with this dataset
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    return clf

# Train SVM model (for demonstration)
clf = train_svm_model()

# Function to extract MFCC features from audio
def extract_mfcc(audio_data, sr):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Average over time axis
    return mfcc

# Function to record and classify audio
def detect_and_classify_noise():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Starting noise detection and classification...")
    
    try:
        while True:
            # Read audio data from the stream
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            audio_data = data.astype(np.float32) / np.max(np.abs(data))  # Normalize audio data
            
            # Calculate volume (RMS)
            volume = np.sqrt(np.mean(np.square(data)))
            
            # If noise detected, proceed with classification
            if volume > THRESHOLD:
                print(f"Unknown noise detected! Volume: {volume:.2f}")
                
                # Extract MFCC features from the audio
                mfcc_features = extract_mfcc(audio_data, RATE)
                
                # Predict the noise class using the trained SVM model
                predicted_class = clf.predict([mfcc_features])
                noise_type = NOISE_CLASSES[predicted_class[0]]
                
                print(f"Noise type: {noise_type}")
            else:
                print(f"Silence. Volume: {volume:.2f}")
            
            # Sleep for a short time to avoid excessive CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping noise detection...")
        stream.stop_stream()
        stream.close()
        p.terminate()

# Run the noise detection and classification
detect_and_classify_noise()
