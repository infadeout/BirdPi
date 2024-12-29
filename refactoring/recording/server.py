import re
from pathlib import Path
from tzlocal import get_localzone
import datetime
import sqlite3
import time
import numpy as np
import librosa
import socket
import threading
import os
import gzip
import logging
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    import tflite_runtime.interpreter as tflite
except BaseException:
    from tensorflow import lite as tflite

# Constants
HEADER = 64
PORT = int(os.getenv('PORT', 5050))
SERVER = os.getenv('SERVER', '0.0.0.0')
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
MODEL_PATH = os.getenv('MODEL_PATH')
LABELS_PATH = os.getenv('LABELS_PATH')
DB_PATH = os.getenv('DB_PATH', '/data/birds.db')

# Global variables
INTERPRETER = None
CLASSES = []
INPUT_LAYER_INDEX = None
OUTPUT_LAYER_INDEX = None
PREDICTED_SPECIES_LIST = []

def splitSignal(sig, rate=48000, seconds=3.0):
    """Split signal into chunks"""
    sig_splits = []
    for i in range(0, len(sig), int(seconds * rate)):
        split = sig[i:i + int(seconds * rate)]
        if len(split) < int(1.5 * rate):
            break
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        sig_splits.append(split)
    return sig_splits

def loadModel():
    """Load TFLite model and labels"""
    global INPUT_LAYER_INDEX, OUTPUT_LAYER_INDEX, CLASSES

    print('LOADING MODEL...', end=' ', flush=True)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debug model input requirements
    input_shape = input_details[0]['shape']
    print(f"Model expects input shape: {input_shape}")

    INPUT_LAYER_INDEX = input_details[0]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']
    
    with open(LABELS_PATH, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]

    print('DONE!')
    return interpreter

def predict(sample, sensitivity=1.0):
    """Analyze audio sample"""
    global INTERPRETER, CLASSES
    
    # Create spectrogram
    spec = librosa.feature.melspectrogram(
        y=sample, 
        sr=48000, 
        n_fft=2048, 
        hop_length=1024, 
        n_mels=40, 
        fmin=50, 
        fmax=15000
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = (spec + 80) / 80
    
    logging.info(f"Spectrogram shape: {spec.shape}")
    
    # Reshape to match model input (1, 144000)
    spec = np.reshape(sample, (1, -1)).astype('float32')
    
    logging.info(f"Final tensor shape: {spec.shape}")
    
    # Make prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, spec)
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid for confidence adjustment
    p_sigmoid = []
    for p in prediction:
        p_sigmoid.append(1.0 / (1.0 + np.exp(-sensitivity * (p - 0.5))))
    
    # Create dictionary mapping species to confidence scores
    p_labels = dict(zip(CLASSES, p_sigmoid))
    
    # Sort by confidence score
    p_sorted = sorted(p_labels.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 10 predictions
    return p_sorted[:10]

def handle_client(conn, addr):
    """Handle client connection"""
    logging.info(f"New connection from {addr}")

    try:
        while True:
            msg = conn.recv(1024).decode(FORMAT)
            if not msg or msg == DISCONNECT_MESSAGE:
                break

            # Parse parameters
            params = msg.split('||')
            audio_path = params[0]
            sensitivity = float(params[4])
            
            logging.info(f"Processing file: {audio_path}")

            # Process audio
            audio, sr = librosa.load(audio_path, sr=48000, mono=True)
            chunks = splitSignal(audio)
            
            logging.info(f"Split audio into {len(chunks)} chunks")

            detections = []
            for i, chunk in enumerate(chunks):
                confidences = predict(chunk, sensitivity)
                logging.info(f"Detected {confidences}
                for i, conf in enumerate(confidences):
                    
                    if conf > 0.1 and i < len(CLASSES):
                        species = CLASSES[i]
                        detections.append((species, conf))
                        logging.info(f"Detected {species} with confidence {conf}")


            # Store results
            conn_db = sqlite3.connect(DB_PATH)
            c = conn_db.cursor()
            timestamp = datetime.datetime.now().isoformat()
            for species, confidence in detections:
                c.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?)", 
                         (timestamp, species, confidence, None, None, None))
            conn_db.commit()
            conn_db.close()
            
            logging.info(f"Stored {len(detections)} detections")
            conn.send("SUCCESS".encode(FORMAT))

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        conn.send("ERROR".encode(FORMAT))
    finally:
        conn.close()
        logging.info(f"Connection closed for {addr}")

def start():
    """Start server"""
    global INTERPRETER
    INTERPRETER = loadModel()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(ADDR)
    except:
        print("Waiting on socket")
        time.sleep(5)
        
    server.listen()
    print(f"Server listening on {SERVER}:{PORT}")
    
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

if __name__ == "__main__":
    start()