import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from typing import Optional
import glob

# Configuration
# Get environment variables or set to default values
RECORDING_LENGTH = int(os.environ.get('RECORDING_LENGTH', 15))
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 48000))
FILENAME_FORMAT = os.environ.get('FILENAME_FORMAT', "{date}-birdnet-{time}.wav")
RECS_DIR = os.environ.get('RECS_DIR', "./recordings")
MAX_RECORDS = int(os.environ.get('MAX_RECORDS', 0))  # 0 means unlimited

def cleanup_old_recordings():
    if MAX_RECORDS <= 0:
        return
        
    # Get list of wav files sorted by creation time
    files = glob.glob(os.path.join(RECS_DIR, "*.wav"))
    files.sort(key=os.path.getctime)
    
    # Remove oldest files if exceed max
    while len(files) >= MAX_RECORDS:
        oldest_file = files.pop(0)
        os.remove(oldest_file)
        print(f"Removed old recording: {oldest_file}")

# Create the directory if it doesn't exist
os.makedirs(RECS_DIR, exist_ok=True)

# Get the number of channels from the default device
device_info = sd.query_devices(sd.default.device, 'input')
CHANNELS = device_info['max_input_channels']

while True:
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

    # Cleanup old recordings before saving new one
    cleanup_old_recordings()
    
    # Save the audio data to a .wav file
    sf.write(os.path.join(RECS_DIR, filename), audio_data, SAMPLE_RATE)
