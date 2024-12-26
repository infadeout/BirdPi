import os
import logging
import socket
import sqlite3
import numpy as np
import librosa
import threading
from datetime import datetime
from pathlib import Path
from tzlocal import get_localzone

# Try to import TFLite runtime, fallback to TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

class BirdNetServer:
    def __init__(self):
        # Server configuration
        self.PORT = int(os.getenv('PORT', 5050))
        self.SERVER = os.getenv('SERVER', '0.0.0.0')
        self.HEADER = 64
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "!DISCONNECT"
        
        # Model paths and settings
        self.MODEL_PATH = os.getenv('MODEL_PATH')
        self.LABELS_PATH = os.getenv('LABELS_PATH')
        self.DB_PATH = os.getenv('DB_PATH')
        self.sample_rate = 48000
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        
        # Initialize components
        self.interpreter = self._load_model()
        self.labels = self._load_labels()
        self._init_database()
        
        # Store model details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_layer_index = input_details[0]['index']
        self.output_layer_index = output_details[0]['index']
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def run(self):
        """Run the server"""
        try:
            self.server_socket.bind((self.SERVER, self.PORT))
            self.server_socket.listen()
            logging.info(f"Server listening on {self.SERVER}:{self.PORT}")
            
            while True:
                conn, addr = self.server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_client, 
                    args=(conn, addr)
                )
                thread.start()
                logging.info(f"Active connections: {threading.active_count() - 1}")
                
        except Exception as e:
            logging.error(f"Server error: {e}")
            raise

    def _handle_client(self, conn, addr):
        """Handle client connection"""
        logging.info(f"New connection from {addr}")
        
        try:
            while True:
                # Receive message
                msg = conn.recv(1024).decode(self.FORMAT)
                if not msg or msg == self.DISCONNECT_MESSAGE:
                    break
                    
                # Parse parameters 
                try:
                    params = msg.split('||')
                    audio_path = params[0]
                    lat = float(params[1])
                    lon = float(params[2]) 
                    week = int(params[3])
                    sensitivity = float(params[4])
                    
                    logging.info(f"Processing file: {audio_path}")
                    
                    # Process audio
                    audio_data = self._load_audio(audio_path)
                    detections = self._analyze_audio(audio_data, sensitivity)
                    
                    # Store results
                    for species, confidence in detections:
                        self._store_detection(species, confidence)
                        
                    # Send response
                    conn.send("SUCCESS".encode(self.FORMAT))
                    
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing message: {e}")
                    conn.send("ERROR".encode(self.FORMAT))
                    
        except Exception as e:
            logging.error(f"Error handling client: {e}")
            conn.send("ERROR".encode(self.FORMAT))
        finally:
            conn.close()

    def _load_model(self):
        """Load TFLite model"""
        logging.info(f"Loading model from {self.MODEL_PATH}")
        interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter

    def _load_labels(self):
        """Load species labels"""
        with open(self.LABELS_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                    (timestamp TEXT, species TEXT, confidence REAL,
                     latitude REAL, longitude REAL, file_name TEXT)''')
        conn.commit()
        conn.close()

    def _load_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Split into 3-second chunks with overlap
        chunks = self._split_signal(audio)
        
        return chunks

    def _split_signal(self, sig, seconds=3.0, overlap=0.0):
        """Split audio signal into chunks"""
        chunks = []
        chunk_samples = int(self.sample_rate * seconds)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        for i in range(0, len(sig), hop_samples):
            chunk = sig[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                # Pad if needed
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
            
        return chunks

    def _analyze_audio(self, chunks, sensitivity=1.0):
        """Analyze audio chunks"""
        detections = []
        
        for chunk in chunks:
            # Create spectrogram
            spec = self._audio_to_spectrogram(chunk)
            
            # Run inference
            self.interpreter.set_tensor(self.input_layer_index, spec)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_layer_index)
            
            # Process predictions
            for i, conf in enumerate(predictions[0]):
                if conf > 0.1:  # Minimum confidence threshold
                    species = self.labels[i]
                    adjusted_conf = self._adjust_confidence(conf, sensitivity)
                    detections.append((species, adjusted_conf))
        
        return detections

    def _audio_to_spectrogram(self, audio):
        """Convert audio to model input format for BirdNET"""
        try:
            import librosa
            
            # Convert to float32 and normalize
            audio = audio.astype('float32')
            audio = audio / np.max(np.abs(audio))

            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=48000,
                n_fft=2048,
                hop_length=1024,
                n_mels=40,
                fmin=50,
                fmax=15000
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec + 80) / 80
            
            # Remove extra dimension - model expects (freq_bins, time_steps)
            mel_spec = mel_spec.astype('float32')
            
            return mel_spec
            
        except Exception as e:
            logging.error(f"Error creating spectrogram: {e}")
            raise

    def _adjust_confidence(self, confidence, sensitivity):
        """Apply sensitivity adjustment to confidence score"""
        return 1.0 / (1.0 + np.exp(-sensitivity * (confidence - 0.5)))

    def _store_detection(self, species, confidence):
        """Store detection in database"""
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?)", 
                 (timestamp, species, confidence, None, None, None))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    server = BirdNetServer()
    server.run()