import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from typing import Optional
import glob
import logging
import sys

# Configuration
# Get environment variables or set to default values
RECORDING_LENGTH = int(os.environ.get('RECORDING_LENGTH', 15))
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 48000))
FILENAME_FORMAT = os.environ.get('FILENAME_FORMAT', "{date}-birdnet-{time}.wav")
RECS_DIR = os.environ.get('RECS_DIR', "./recordings")
MAX_RECORDS = int(os.environ.get('MAX_RECORDS', 0))  # 0 means unlimited

# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

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

# List all ALSA devices
devices = sd.query_devices()
logging.info(f"Available audio devices on Raspberry Pi:\n{devices}")
# Get the number of channels from the default device
device_info = sd.query_devices(sd.default.device, 'input')
logging.info(f"Using input device:\n{device_info}")
CHANNELS = device_info['max_input_channels']

while True:
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            logging.warning(f"Stream callback status: {status}")
        vol_norm = np.linalg.norm(indata) * 10
        print("|" * int(vol_norm))
        audio_data.append(indata.copy())

    logging.info("Starting new recording session...")
    with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=SAMPLE_RATE) as stream:
        stream.start()
        logging.info(f"Recording for {RECORDING_LENGTH} seconds...")
        sd.sleep(RECORDING_LENGTH * 1000)

    audio_data = np.concatenate(audio_data)
    filename = FILENAME_FORMAT.format(
        date=datetime.now().strftime("%Y-%m-%d"), 
        time=datetime.now().strftime("%H-%M-%S")
    )
    
    logging.info("Cleaning up old recordings...")
    cleanup_old_recordings()
    
    output_path = os.path.join(RECS_DIR, filename)
    logging.info(f"Saving recording to {output_path}")
    sf.write(output_path, audio_data, SAMPLE_RATE, format='WAV')
    logging.info("Recording saved successfully")
