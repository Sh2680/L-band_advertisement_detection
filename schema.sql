-- ============================================
-- L-BAND ADS DETECTION SYSTEM - DATABASE SCHEMA
-- ============================================

CREATE DATABASE IF NOT EXISTS lband_ads
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE lband_ads;


CREATE TABLE IF NOT EXISTS lband_ad_events (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique event ID',
    track_id INT NOT NULL COMMENT 'DeepSORT tracking ID',
    start_time DATETIME NOT NULL COMMENT 'When L-band first appeared',
    end_time DATETIME NOT NULL COMMENT 'When L-band disappeared',
    duration_seconds FLOAT NOT NULL COMMENT 'Total duration in seconds',
    appearance_count INT NOT NULL DEFAULT 1 COMMENT 'Number of frames L-band appeared in',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Record creation timestamp',
    
    -- Indexes for performance
    INDEX idx_track_id (track_id),
    INDEX idx_start_time (start_time),
    INDEX idx_created_at (created_at),
    INDEX idx_duration (duration_seconds)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Stores L-band advertisement detection events';






