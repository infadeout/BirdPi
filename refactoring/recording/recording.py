import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime

# Configuration
RECORDING_LENGTH = 15  # seconds
SAMPLE_RATE = 48000  # sample rate
FILENAME_FORMAT = "{date}-birdnet-{time}.wav"
RECS_DIR = "./recordings"  # replace with your directory

# Create the directory if it doesn't exist
os.makedirs(RECS_DIR, exist_ok=True)

# Get the number of channels from the default device
device_info = sd.query_devices(sd.default.device, 'input')
CHANNELS = device_info['max_input_channels']

# List to hold audio data
audio_data = []

# Callback function for the stream
def callback(indata, frames, time, status):
    # This function will be called by the stream for every chunk of audio
    vol_norm = np.linalg.norm(indata) * 10
    print("|" * int(vol_norm))  # Print a simple visualization of the volume
    audio_data.append(indata.copy())  # Append the audio data to the list

# Create a stream
with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=SAMPLE_RATE) as stream:
    # Start the stream
    stream.start()

    # Record for the specified duration
    sd.sleep(RECORDING_LENGTH * 1000)

# Convert the list of arrays into a single array
audio_data = np.concatenate(audio_data)

# Generate the filename
filename = FILENAME_FORMAT.format(date=datetime.now().strftime("%Y-%m-%d"), time=datetime.now().strftime("%H-%M-%S"))

# Save the audio data to a .wav file
sf.write(os.path.join(RECS_DIR, filename), audio_data, SAMPLE_RATE)