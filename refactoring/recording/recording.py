import sounddevice as sd
import numpy as np
import os
from datetime import datetime

# Configuration
RECORDING_LENGTH = 15  # seconds
SAMPLE_RATE = 48000  # sample rate
CHANNELS = 2  # number of channels
FILENAME_FORMAT = "{date}-birdnet-{time}.wav"
RECS_DIR = "/path/to/recordings"  # replace with your directory

# Create the directory if it doesn't exist
os.makedirs(RECS_DIR, exist_ok=True)

# Callback function for the stream
def callback(indata, frames, time, status):
    # This function will be called by the stream for every chunk of audio
    vol_norm = np.linalg.norm(indata) * 10
    print("|" * int(vol_norm))  # Print a simple visualization of the volume

# Create a stream
stream = sd.InputStream(callback=callback, channels=CHANNELS, samplerate=SAMPLE_RATE)

# Start the stream
stream.start()

# Record for the specified duration
sd.sleep(RECORDING_LENGTH * 1000)

# Stop the stream
stream.stop()

# Get the data from the stream
recording = stream.read(RECORDING_LENGTH * SAMPLE_RATE)

# Generate the filename
now = datetime.now()
filename = FILENAME_FORMAT.format(date=now.strftime("%F"), time=now.strftime("%H:%M:%S"))
filepath = os.path.join(RECS_DIR, filename)

# Save the data to a WAV file
sd.write(filepath, recording, SAMPLE_RATE)

print(f"Recording saved to {filepath}")