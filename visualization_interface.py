"""
ATLAS Football - Visualization and Dashboard Interface

This module creates visualizations and dashboard interfaces for displaying
ATLAS analysis results to coaches, players, and medical staff.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atlas_visualization")


class VisualizationGenerator:
    """Generates visualizations for ATLAS analysis results."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.color_scheme = {
            "primary": "#0052CC",
            "secondary": "#00B8D9",
            "success": "#36B37E",
            "warning": "#FFAB00",
            "danger": "#FF5630",
            "info": "#6554C0",
            "text": "#172B4D",
            "background": "#F4F5F7"
        }
        logger.info(f"Initialized visualization generator with output directory: {output_dir}")
    
    def generate_movement_visualization(self, 
                                       sensor_data_sequence: List[Dict],
                                       movement_type: str = None,
                                       output_filename: str = None) -> str:
        """
        Generate visualization of a movement sequence.
        
        Args:
            sensor_data_sequence: List of sensor data frames
            movement_type: Type of movement for labeling
            output_filename: Name for output file (optional)
            
        Returns:
            Path to saved visualization file
        """
        # Create a multi-panel visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Set title based on movement type
        title = f"{movement_type.title()} Movement Analysis" if movement_type else "Movement Analysis"
        fig.suptitle(title, fontsize=16)
        
        # Preprocess data for plotting
        timestamps = []
        accel_data = []
        gyro_data = []
        orientation_data = []
        
        for frame in sensor_data_sequence:
            timestamps.append(frame.get("timestamp", 0))
            
            # Get first available IMU data
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                if "filtered" in sensor_data and "acceleration" in sensor_data["filtered"]:
                    accel_data.append(sensor_data["filtered"]["acceleration"])
                    
                if "filtered" in sensor_data and "angular_velocity" in sensor_data["filtered"]:
                    gyro_data.append(sensor_data["filtered"]["angular_velocity"])
                    
                if "orientation" in sensor_data:
                    orientation_data.append([
                        sensor_data["orientation"]["roll"],
                        sensor_data["orientation"]["pitch"],
                        sensor_data["orientation"]["yaw"]
                    ])
                    
                # Just use first sensor for this visualization
                break
        
        # Create relative timestamps (starting at 0)
        if timestamps:
            relative_times = [t - timestamps[0] for t in timestamps]
        else:
            relative_times = list(range(len(sensor_data_sequence)))
        
        # Panel 1: Acceleration
        if accel_data:
            accel_data = np.array(accel_data)
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(relative_times, accel_data[:, 0], label='X', color='r')
            ax1.plot(relative_times, accel_data[:, 1], label='Y', color='g')
            ax1.plot(relative_times, accel_data[:, 2], label='Z', color='b')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.set_title('Acceleration')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Panel 2: Angular Velocity
        if gyro_data:
            gyro_data = np.array(gyro_data)
            ax2 = fig.add_subplot(3, 1, 2)
            ax2.plot(relative_times, gyro_data[:, 0], label='X', color='r')
            ax2.plot(relative_times, gyro_data[:, 1], label='Y', color='g')
            ax2.plot(relative_times, gyro_data[:, 2], label='Z', color='b')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.set_title('Angular Velocity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Orientation
        if orientation_data:
            orientation_data = np.array(orientation_data)
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(relative_times,
