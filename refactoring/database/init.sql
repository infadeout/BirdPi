-- Create tables for bird detections
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    species TEXT NOT NULL,
    confidence REAL NOT NULL,
    audio_file TEXT,
    latitude REAL,
    longitude REAL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_detections_species ON detections(species);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);

-- Create views
CREATE VIEW IF NOT EXISTS daily_detections AS
SELECT DATE(timestamp) as date, 
       species,
       COUNT(*) as count,
       AVG(confidence) as avg_confidence
FROM detections 
GROUP BY DATE(timestamp), species;