import os
import time
import glob
import logging
import subprocess
import socket
from datetime import datetime
from pathlib import Path
import numpy as np

class BirdNetAnalyzer:
    def __init__(self):
        # Configuration from environment variables
        self.recs_dir = os.getenv('RECS_DIR', './recordings')
        self.processed_dir = os.getenv('PROCESSED_DIR', './processed')
        self.include_list = os.getenv('INCLUDE_LIST', '')
        self.exclude_list = os.getenv('EXCLUDE_LIST', '')
        self.latitude = float(os.getenv('LATITUDE', 0))
        self.longitude = float(os.getenv('LONGITUDE', 0))
        self.sensitivity = float(os.getenv('SENSITIVITY', 1.0))
        self.overlap = float(os.getenv('OVERLAP', 0.0))
        self.min_conf = float(os.getenv('MIN_CONF', 0.1))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        
        self.server_address = ('server', 5050)
        
    def calculate_week(self):
        """Calculate week number (1-48) from current date"""
        now = datetime.now()
        week_of_year = (now.month - 1) * 4
        day_of_month = now.day
        
        if day_of_month <= 7:
            week = week_of_year + 1
        elif day_of_month <= 14:
            week = week_of_year + 2
        elif day_of_month <= 21:
            week = week_of_year + 3
        else:
            week = week_of_year + 4
            
        return min(max(1, week), 48)

    def wait_for_server(self):
        """Wait for BirdNET analysis server to be ready"""
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(self.server_address)
                    logging.info("Connected to analysis server")
                    break
            except socket.error:
                logging.info("Waiting for analysis server...")
                time.sleep(5)

    def get_audio_files(self, directory):
        """Get list of WAV files to process"""
        wav_files = glob.glob(os.path.join(directory, "*.wav"))
        return sorted([f for f in wav_files if os.path.getsize(f) > 0])[:20]

    def move_analyzed_files(self, files, analyzed_dir):
        """Move processed files to analyzed directory"""
        os.makedirs(analyzed_dir, exist_ok=True)
        for wav_file in files:
            csv_file = wav_file + '.csv'
            if os.path.exists(csv_file):
                os.rename(wav_file, os.path.join(analyzed_dir, os.path.basename(wav_file)))
                os.rename(csv_file, os.path.join(analyzed_dir, os.path.basename(csv_file)))

    def analyze_file(self, audio_file):
        """Run BirdNET analysis on a single file"""
        try:
            week = self.calculate_week()
            logging.info(f"Analyzing file: {audio_file}")
            logging.info(f"Connecting to server at {self.server_address}")  
            # Connect to server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.server_address)
                logging.info(f"Connected to server at {self.server_address}")
                
                # Send parameters
                params = f"{audio_file}||{self.latitude}||{self.longitude}||{week}||{self.sensitivity}"
                s.send(params.encode('utf-8'))
                logging.info("Sent analysis request")
                
                # Get response
                response = s.recv(1024).decode('utf-8')
                logging.info(f"Server response: {response}")
                
                return response == "SUCCESS"
                
        except Exception as e:
            logging.error(f"Analysis failed for {audio_file}: {e}")
            return False

    def run(self):
        """Main analysis loop"""
        while True:
            try:
                files = self.get_audio_files(self.recs_dir)
                if files:
                    logging.info(f"Found {len(files)} files to analyze")
                    analyzed_files = []
                    
                    for f in files:
                        if self.analyze_file(f):
                            logging.info(f"Successfully analyzed {f}")
                            analyzed_files.append(f)
                        else:
                            logging.error(f"Failed to analyze {f}")
                    
                    if analyzed_files:
                        analyzed_dir = os.path.join(self.processed_dir, "Analyzed")
                        self.move_analyzed_files(analyzed_files, analyzed_dir)
                        
                time.sleep(3)
                
            except Exception as e:
                logging.error(f"Error in analysis loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    analyzer = BirdNetAnalyzer()
    analyzer.run()