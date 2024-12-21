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
        
    def analyze_audio(self, audio_data, sensitivity=1.0):
        """Analyze audio data using BirdNET model"""
        # Preprocess audio
        specs = self._audio_to_spectrogram(audio_data)
        
        # Run inference
        self.interpreter.set_tensor(self.input_layer_index, specs)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_layer_index)
        
        # Process results
        detections = []
        for i, conf in enumerate(predictions[0]):
            if conf > 0.1:  # Minimum confidence threshold
                species = self.labels[i]
                adjusted_conf = self._adjust_confidence(conf, sensitivity)
                detections.append((species, adjusted_conf))
                
        return detections
        
    def _audio_to_spectrogram(self, audio):
        """Convert audio to model input format"""
        # Implementation depends on specific model requirements
        pass
        
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
                self._handle_client(conn)
            except Exception as e:
                logging.error(f"Error handling client: {e}")
            finally:
                conn.close()

if __name__ == "__main__":
    server = BirdNetServer()
    server.run()