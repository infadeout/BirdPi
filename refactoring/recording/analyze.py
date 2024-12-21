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
        
        self.server_address = ('localhost', 5050)
        
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
        week = self.calculate_week()
        
        cmd = [
            'analyze.py',
            '--i', audio_file,
            '--o', f'{audio_file}.csv',
            '--lat', str(self.latitude),
            '--lon', str(self.longitude),
            '--week', str(week),
            '--sensitivity', str(self.sensitivity),
            '--overlap', str(self.overlap),
            '--min_conf', str(self.min_conf)
        ]

        if os.path.exists(self.include_list):
            cmd.extend(['--include_list', self.include_list])
        if os.path.exists(self.exclude_list):
            cmd.extend(['--exclude_list', self.exclude_list])

        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Analyzed {audio_file}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Analysis failed for {audio_file}: {e}")

    def run(self):
        """Main analysis loop"""
        self.wait_for_server()

        while True:
            # Check StreamData directory
            stream_dir = os.path.join(self.recs_dir, "StreamData")
            if os.path.exists(stream_dir):
                files = self.get_audio_files(stream_dir)
                for f in files:
                    self.analyze_file(f)
                self.move_analyzed_files(files, os.path.join(stream_dir, "Analyzed"))

            # Check today's directory
            today_dir = os.path.join(
                self.recs_dir,
                datetime.now().strftime("%B-%Y/%d-%A")
            )
            if os.path.exists(today_dir):
                files = self.get_audio_files(today_dir) 
                for f in files:
                    self.analyze_file(f)
                self.move_analyzed_files(files, os.path.join(today_dir, "Analyzed"))

            time.sleep(3)

if __name__ == "__main__":
    analyzer = BirdNetAnalyzer()
    analyzer.run()