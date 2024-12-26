import os
import logging
import socket
import sqlite3
import numpy as np
import tflite_runtime.interpreter as tflite
from datetime import datetime

class BirdNetServer:
    def __init__(self):
        # Configuration
        self.PORT = int(os.getenv('PORT', 5050))
        self.SERVER = os.getenv('SERVER', 'localhost')
        self.MODEL_PATH = os.getenv('MODEL_PATH')
        self.LABELS_PATH = os.getenv('LABELS_PATH')
        self.DB_PATH = os.getenv('DB_PATH')
        self.sample_rate = 48000  # Add sample rate
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        
        # Load model and labels
        self.interpreter = self._load_model()
        self.labels = self._load_labels()
        
        # Get input/output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_layer_index = input_details[0]['index']
        self.output_layer_index = output_details[0]['index']
        
        # Initialize database
        self._init_database()

    def _load_audio(self, audio_path):
        """Load and preprocess audio file"""
        import soundfile as sf
        
        try:
            # Load audio file
            audio, sr = sf.read(audio_path)
            logging.info(f"Loaded audio file: {audio_path}, sample rate: {sr}")
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logging.error(f"Error loading audio file {audio_path}: {e}")
            raise

    def _handle_client(self, conn, addr):
        """Handle client connection and process audio file"""
        try:
            # Receive data
            data = conn.recv(1024).decode('utf-8')
            params = data.split('||')
            
            audio_path = params[0]
            lat = float(params[1])
            lon = float(params[2])
            week = int(params[3])
            sensitivity = float(params[4])
            
            logging.info(f"Processing file: {audio_path}")
            
            # Process audio through model
            audio_data = self._load_audio(audio_path)
            predictions = self._analyze_audio(audio_data, sensitivity)
            
            # Store results
            for species, confidence in predictions:
                self._store_detection(species, confidence, audio_path, lat, lon)
            
            # Send success response
            conn.send("SUCCESS".encode('utf-8'))
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            conn.send("ERROR".encode('utf-8'))
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
                    (timestamp TEXT, species TEXT, confidence REAL)''')
        conn.commit()
        conn.close()
        
    def _analyze_audio(self, audio_data, sensitivity=1.0):
        """Analyze audio data using BirdNET model"""
        try:
            # Get model input shape
            input_details = self.interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            
            # Preprocess audio to spectrograms
            specs = self._audio_to_spectrogram(audio_data)
            
            detections = []
            for spec in specs:
                # Ensure correct shape (1, freq_bins, time_steps)
                spec = np.reshape(spec, input_shape)
                
                # Run inference
                self.interpreter.set_tensor(self.input_layer_index, spec)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_layer_index)
                
                # Process results
                for i, conf in enumerate(predictions[0]):
                    if conf > 0.1:  # Minimum confidence threshold
                        species = self.labels[i]
                        adjusted_conf = self._adjust_confidence(conf, sensitivity)
                        detections.append((species, adjusted_conf))
                        
            return detections
            
        except Exception as e:
            logging.error(f"Error analyzing audio: {e}")
            raise
            
    def _audio_to_spectrogram(self, audio):
        """Convert audio to model input format for BirdNET"""
        try:
            import librosa
            
            # Convert to float32 and normalize
            audio = audio.astype('float32')
            audio = audio / np.max(np.abs(audio))
            
            # Ensure audio is at 48kHz
            if self.sample_rate != 48000:
                audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=48000)
            
            # Split into 3-second chunks
            sig_splits = []
            samples_per_chunk = 48000 * 3  # 3 seconds at 48kHz
            
            for i in range(0, len(audio), samples_per_chunk):
                split = audio[i:i + samples_per_chunk]
                
                # Skip if less than 1.5 seconds
                if len(split) < int(48000 * 1.5):
                    continue
                    
                # Pad if needed
                if len(split) < samples_per_chunk:
                    split = np.pad(split, (0, samples_per_chunk - len(split)))
                    
                # Create mel spectrogram
                spec = librosa.feature.melspectrogram(
                    y=split,
                    sr=48000,
                    n_fft=2048,
                    hop_length=1024,
                    n_mels=40,
                    fmin=50,
                    fmax=15000
                )
                
                # Convert to log scale and normalize
                spec = librosa.power_to_db(spec, ref=np.max)
                spec = (spec + 80) / 80
                
                # Add batch dimension
                spec = np.expand_dims(spec, 0)
                
                sig_splits.append(spec.astype('float32'))
                
            return sig_splits
            
        except Exception as e:
            logging.error(f"Error creating spectrogram: {e}")
            raise
        
    def _adjust_confidence(self, confidence, sensitivity):
        """Apply sensitivity adjustment to confidence scores"""
        return 1.0 / (1.0 + np.exp(-sensitivity * (confidence - 0.5)))
        
    def store_detection(self, species, confidence):
        """Store detection in database"""
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute("INSERT INTO detections VALUES (?, ?, ?)", 
                 (timestamp, species, confidence))
        conn.commit()
        conn.close()

    def run(self):
        """Run the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.SERVER, self.PORT))
        server_socket.listen()
        
        logging.info(f"Server listening on {self.SERVER}:{self.PORT}")
        
        while True:
            conn, addr = server_socket.accept()
            logging.info(f"New connection from {addr}")
            
            try:
                self._handle_client(conn, addr)
            except Exception as e:
                logging.error(f"Error handling client: {e}")
            finally:
                conn.close()

if __name__ == "__main__":
    server = BirdNetServer()
    server.run()