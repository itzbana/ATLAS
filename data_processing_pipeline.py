"""
ATLAS Football - Data Processing Pipeline

This module processes raw sensor data from football players,
performs cleaning, synchronization, and feature extraction.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atlas_data_processing")


class DataProcessor:
    """Base class for processing sensor data."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def process(self, data: Dict) -> Dict:
        """Process the data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process method")


class IMUProcessor(DataProcessor):
    """Processes raw IMU data."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Default filter parameters
        self.filter_params = self.config.get("filter_params", {
            "cutoff": 20,  # Hz
            "fs": 100,     # Hz (sampling frequency)
            "order": 4     # Filter order
        })
    
    def _apply_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply a low-pass filter to remove high-frequency noise."""
        cutoff = self.filter_params["cutoff"]
        fs = self.filter_params["fs"]
        order = self.filter_params["order"]
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Create the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def _calculate_orientation(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Calculate orientation from IMU data using complementary filter.
        This is a simplified implementation - production code would use more sophisticated algorithms.
        """
        # This is a placeholder for a proper sensor fusion algorithm
        # A real implementation would use Mahony, Madgwick, or Kalman filtering
        
        # Normalize acceleration to get gravity direction
        acc_norm = np.linalg.norm(accel)
        if acc_norm > 0:
            accel = accel / acc_norm
        
        # Simplified calculation of roll and pitch from accelerometer
        roll = np.arctan2(accel[1], accel[2])
        pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        
        # For a proper implementation, we would integrate gyroscope and correct with magnetometer
        # This is just a placeholder
        yaw = np.arctan2(mag[1], mag[0])
        
        return np.array([roll, pitch, yaw])
    
    def _calculate_jerk(self, accel: np.ndarray, dt: float) -> np.ndarray:
        """Calculate jerk (rate of change of acceleration)."""
        # Simple finite difference method
        jerk = np.diff(accel, axis=0) / dt
        # Pad to maintain original length
        jerk = np.vstack([jerk, jerk[-1]])
        return jerk
    
    def process(self, imu_data: Dict) -> Dict:
        """Process IMU data to extract features."""
        # Extract raw IMU data
        accel = np.array(imu_data["acceleration"])
        gyro = np.array(imu_data["angular_velocity"])
        mag = np.array(imu_data["magnetic_field"])
        
        # Apply filtering
        filtered_accel = self._apply_lowpass_filter(accel)
        filtered_gyro = self._apply_lowpass_filter(gyro)
        
        # Calculate derivative metrics
        dt = 1.0 / self.filter_params["fs"]  # Time step based on sampling rate
        jerk = self._calculate_jerk(filtered_accel, dt)
        
        # Calculate orientation
        orientation = self._calculate_orientation(filtered_accel, filtered_gyro, mag)
        
        # Calculate movement intensity metrics
        accel_magnitude = np.linalg.norm(filtered_accel)
        gyro_magnitude = np.linalg.norm(filtered_gyro)
        jerk_magnitude = np.linalg.norm(jerk)
        
        # Return processed data with extracted features
        processed_data = {
            "timestamp": imu_data["timestamp"],
            "device_id": imu_data["device_id"],
            "location": imu_data["location"],
            "raw": {
                "acceleration": accel.tolist(),
                "angular_velocity": gyro.tolist(),
                "magnetic_field": mag.tolist()
            },
            "filtered": {
                "acceleration": filtered_accel.tolist(),
                "angular_velocity": filtered_gyro.tolist()
            },
            "derived": {
                "jerk": jerk.tolist()
            },
            "orientation": {
                "roll": float(orientation[0]),
                "pitch": float(orientation[1]),
                "yaw": float(orientation[2])
            },
            "magnitudes": {
                "acceleration": float(accel_magnitude),
                "angular_velocity": float(gyro_magnitude),
                "jerk": float(jerk_magnitude)
            }
        }
        
        return processed_data


class ForceplateProcessor(DataProcessor):
    """Processes raw force plate data."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Default filter parameters
        self.filter_params = self.config.get("filter_params", {
            "cutoff": 50,  # Hz
            "fs": 1000,    # Hz (sampling frequency)
            "order": 4     # Filter order
        })
    
    def _apply_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply a low-pass filter to remove high-frequency noise."""
        cutoff = self.filter_params["cutoff"]
        fs = self.filter_params["fs"]
        order = self.filter_params["order"]
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Create the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply the filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def _calculate_impulse(self, force: np.ndarray, dt: float) -> float:
        """Calculate impulse (time integral of force)."""
        # Simple trapezoidal integration
        return float(np.trapz(force, dx=dt))
    
    def _calculate_power(self, force: np.ndarray, cop_velocity: np.ndarray) -> float:
        """Calculate power (force * velocity)."""
        # For a full implementation, we would need displacement data
        # This is a simplified placeholder
        return float(np.dot(force[:2], cop_velocity))
    
    def process(self, fp_data: Dict) -> Dict:
        """Process force plate data to extract features."""
        # Extract raw force plate data
        force = np.array(fp_data["force"])
        cop = np.array(fp_data["center_of_pressure"])
        
        # Apply filtering
        filtered_force = self._apply_lowpass_filter(force)
        
        # Calculate force metrics
        force_magnitude = np.linalg.norm(filtered_force)
        vertical_force = filtered_force[2]  # Assuming z is vertical
        
        # For a full implementation, we would analyze CoP trajectory
        # For now, we just pass it through
        
        # Calculate impulse (simplified)
        dt = 1.0 / self.filter_params["fs"]  # Time step based on sampling rate
        impulse = self._calculate_impulse(vertical_force, dt)
        
        # In a real implementation, we would calculate:
        # - Ground contact time
        # - Rate of force development
        # - Power from force and velocity
        # - Asymmetry metrics
        # - etc.
        
        # Return processed data with extracted features
        processed_data = {
            "timestamp": fp_data["timestamp"],
            "device_id": fp_data["device_id"],
            "raw": {
                "force": force.tolist(),
                "center_of_pressure": cop.tolist()
            },
            "filtered": {
                "force": filtered_force.tolist()
            },
            "metrics": {
                "force_magnitude": float(force_magnitude),
                "vertical_force": float(vertical_force),
                "impulse": impulse
            }
        }
        
        return processed_data


class PlayerDataProcessor:
    """Processes all sensor data for a single player."""
    
    def __init__(self, player_id: str, position: str):
        self.player_id = player_id
        self.position = position
        self.processors = {
            "IMU": IMUProcessor(),
            "ForcePlace": ForceplateProcessor()
        }
        logger.info(f"Initialized data processor for player {player_id}")
    
    def process_data_point(self, data_point: Dict) -> Dict:
        """Process a data point containing multiple sensor readings."""
        processed_point = {
            "timestamp": data_point["timestamp"],
            "session_id": data_point["session_id"],
            "player_id": self.player_id,
            "position": self.position,
            "processed_sensor_data": {}
        }
        
        # Process each sensor's data
        for sensor_key, sensor_data in data_point["sensor_data"].items():
            # Extract sensor type from key (e.g., "IMU_H001" -> "IMU")
            sensor_type = sensor_key.split("_")[0]
            
            if sensor_type in self.processors:
                try:
                    processed_data = self.processors[sensor_type].process(sensor_data)
                    processed_point["processed_sensor_data"][sensor_key] = processed_data
                except Exception as e:
                    logger.error(f"Error processing {sensor_type} data: {e}")
                    processed_point["processed_sensor_data"][sensor_key] = {"error": str(e)}
            else:
                logger.warning(f"No processor found for sensor type: {sensor_type}")
        
        return processed_point


class DataPipeline:
    """Main data processing pipeline for ATLAS system."""
    
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        self.player_processors = {}
        self.output_path = self.config.get("output_path", "processed_data")
        os.makedirs(self.output_path, exist_ok=True)
        
        logger.info(f"Initialized data pipeline with output to {self.output_path}")
    
    def add_player(self, player_id: str, position: str) -> None:
        """Add a player to the processing pipeline."""
        self.player_processors[player_id] = PlayerDataProcessor(player_id, position)
        logger.info(f"Added player {player_id} to data pipeline")
    
    def process_session_data(self, session_data_path: str) -> str:
        """
        Process all data files for a session.
        
        Args:
            session_data_path: Path to directory containing session data files
            
        Returns:
            Path to output directory with processed data
        """
        session_path = Path(session_data_path)
        if not session_path.exists() or not session_path.is_dir():
            raise ValueError(f"Invalid session data path: {session_data_path}")
        
        # Extract session ID from directory name or first file
        first_file = next(session_path.glob("*.json"), None)
        if not first_file:
            raise ValueError(f"No data files found in {session_data_path}")
        
        with open(first_file, 'r') as f:
            first_data = json.load(f)
            # Get session ID from first data point
            session_id = first_data[0]["session_id"] if isinstance(first_data, list) else first_data["session_id"]
        
        # Create output directory for processed data
        output_dir = Path(self.output_path) / session_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each data file
        files_processed = 0
        for data_file in session_path.glob("*.json"):
            logger.info(f"Processing file: {data_file}")
            
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # If data is a list of data points
                if isinstance(data, list):
                    processed_data = []
                    for data_point in data:
                        player_id = data_point["player_id"]
                        # Add player processor if not exists
                        if player_id not in self.player_processors:
                            position = data_point.get("position", "unknown")
                            self.add_player(player_id, position)
                        
                        processed_point = self.player_processors[player_id].process_data_point(data_point)
                        processed_data.append(processed_point)
                    
                    # Save processed data
                    output_file = output_dir / f"processed_{data_file.name}"
                    with open(output_file, 'w') as f:
                        json.dump(processed_data, f)
                    
                else:  # Single data point
                    player_id = data["player_id"]
                    # Add player processor if not exists
                    if player_id not in self.player_processors:
                        position = data.get("position", "unknown")
                        self.add_player(player_id, position)
                    
                    processed_data = self.player_processors[player_id].process_data_point(data)
                    
                    # Save processed data
                    output_file = output_dir / f"processed_{data_file.name}"
                    with open(output_file, 'w') as f:
                        json.dump(processed_data, f)
                
                files_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing file {data_file}: {e}")
        
        logger.info(f"Processed {files_processed} files for session {session_id}")
        return
