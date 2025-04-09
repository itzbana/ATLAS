"""
ATLAS Football - Biomechanical Modeling Module

This module handles the creation of player-specific biomechanical models
and analyzes player movement techniques for football-specific actions.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atlas_biomechanical")


class AnatomicalModel:
    """Base class for anatomical models."""
    
    def __init__(self, model_name: str, player_id: str = None):
        self.model_name = model_name
        self.player_id = player_id
        self.segments = {}
        self.joints = {}
        self.anthropometrics = {}
        logger.info(f"Initialized anatomical model: {model_name}")
    
    def add_segment(self, segment_id: str, segment_data: Dict) -> None:
        """Add a body segment to the model."""
        self.segments[segment_id] = segment_data
    
    def add_joint(self, joint_id: str, joint_data: Dict) -> None:
        """Add a joint to the model."""
        self.joints[joint_id] = joint_data
    
    def set_anthropometrics(self, anthropometrics: Dict) -> None:
        """Set anthropometric measurements for the model."""
        self.anthropometrics = anthropometrics
    
    def save_model(self, output_path: str) -> str:
        """Save the model to file."""
        os.makedirs(output_path, exist_ok=True)
        
        model_data = {
            "model_name": self.model_name,
            "player_id": self.player_id,
            "segments": self.segments,
            "joints": self.joints,
            "anthropometrics": self.anthropometrics
        }
        
        filename = os.path.join(output_path, f"{self.model_name}_{self.player_id}.json")
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Saved anatomical model to {filename}")
        return filename
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load a model from file."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        model = cls(model_data["model_name"], model_data["player_id"])
        model.segments = model_data["segments"]
        model.joints = model_data["joints"]
        model.anthropometrics = model_data["anthropometrics"]
        
        logger.info(f"Loaded anatomical model from {model_path}")
        return model


class FootballPlayerModel(AnatomicalModel):
    """Specialized anatomical model for football players."""
    
    def __init__(self, player_id: str, position: str):
        model_name = f"football_{position.lower()}"
        super().__init__(model_name, player_id)
        self.position = position
        self.performance_metrics = {}
        self.risk_factors = {}
        
        # Initialize with default segment definitions based on position
        self._initialize_position_specific_model()
    
    def _initialize_position_specific_model(self) -> None:
        """Initialize segments and joints based on player position."""
        # All positions share these basic segments
        self.add_segment("pelvis", {
            "mass_percentage": 0.142,  # Percentage of total body mass
            "length_percentage": 0.13,  # Percentage of height
            "sensors": ["pelvis_imu"]
        })
        
        self.add_segment("torso", {
            "mass_percentage": 0.355,
            "length_percentage": 0.30,
            "sensors": ["upper_back_imu", "lower_back_imu"]
        })
        
        self.add_segment("head", {
            "mass_percentage": 0.081,
            "length_percentage": 0.14,
            "sensors": ["helmet_imu"]
        })
        
        # Add upper and lower limbs
        for side in ["left", "right"]:
            self.add_segment(f"{side}_upper_arm", {
                "mass_percentage": 0.028,
                "length_percentage": 0.17,
                "sensors": [f"{side}_upper_arm_imu"]
            })
            
            self.add_segment(f"{side}_forearm", {
                "mass_percentage": 0.016,
                "length_percentage": 0.15,
                "sensors": [f"{side}_forearm_imu"]
            })
            
            self.add_segment(f"{side}_hand", {
                "mass_percentage": 0.006,
                "length_percentage": 0.05,
                "sensors": []  # Typically no sensors here
            })
            
            self.add_segment(f"{side}_thigh", {
                "mass_percentage": 0.10,
                "length_percentage": 0.24,
                "sensors": [f"{side}_thigh_imu"]
            })
            
            self.add_segment(f"{side}_shank", {
                "mass_percentage": 0.047,
                "length_percentage": 0.24,
                "sensors": [f"{side}_shank_imu"]
            })
            
            self.add_segment(f"{side}_foot", {
                "mass_percentage": 0.014,
                "length_percentage": 0.15,
                "sensors": [f"{side}_foot_imu"]
            })
        
        # Add position-specific customizations
        if self.position.lower() == "quarterback":
            # Adjust throwing arm parameters
            throwing_arm = "right"  # Default, should be set based on player data
            self.segments[f"{throwing_arm}_upper_arm"]["importance"] = "critical"
            self.segments[f"{throwing_arm}_forearm"]["importance"] = "critical"
            
            # Add QB-specific performance metrics
            self.performance_metrics["throwing_mechanics"] = {
                "arm_angle": None,
                "torso_rotation": None,
                "follow_through": None,
                "release_timing": None
            }
            
        elif self.position.lower() in ["offensive_lineman", "defensive_lineman"]:
            # Adjust for linemen
            self.segments["torso"]["mass_percentage"] *= 1.1  # Linemen typically have larger torsos
            
            # Add linemen-specific performance metrics
            self.performance_metrics["blocking_mechanics"] = {
                "stance_width": None,
                "leverage": None,
                "hand_placement": None,
                "drive_power": None
            }
        
        # Add basic joints
        self._initialize_joints()
    
    def _initialize_joints(self) -> None:
        """Initialize the joint definitions."""
        # Define the key joints with their connecting segments
        self.add_joint("neck", {
            "connects": ["head", "torso"],
            "dofs": 3,  # Degrees of freedom
            "range_of_motion": {
                "flexion": [-60, 70],  # Degrees
                "lateral_bend": [-40, 40],
                "rotation": [-80, 80]
            }
        })
        
        self.add_joint("lower_back", {
            "connects": ["pelvis", "torso"],
            "dofs": 3,
            "range_of_motion": {
                "flexion": [-30, 90],
                "lateral_bend": [-35, 35],
                "rotation": [-45, 45]
            }
        })
        
        # Add shoulder, elbow, wrist, hip, knee, ankle joints
        for side in ["left", "right"]:
            self.add_joint(f"{side}_shoulder", {
                "connects": ["torso", f"{side}_upper_arm"],
                "dofs": 3,
                "range_of_motion": {
                    "flexion": [-60, 180],
                    "abduction": [0, 180],
                    "rotation": [-90, 90]
                }
            })
            
            self.add_joint(f"{side}_elbow", {
                "connects": [f"{side}_upper_arm", f"{side}_forearm"],
                "dofs": 2,
                "range_of_motion": {
                    "flexion": [0, 150],
                    "rotation": [-90, 90]
                }
            })
            
            self.add_joint(f"{side}_wrist", {
                "connects": [f"{side}_forearm", f"{side}_hand"],
                "dofs": 2,
                "range_of_motion": {
                    "flexion": [-80, 80],
                    "deviation": [-20, 20]
                }
            })
            
            self.add_joint(f"{side}_hip", {
                "connects": ["pelvis", f"{side}_thigh"],
                "dofs": 3,
                "range_of_motion": {
                    "flexion": [-30, 120],
                    "abduction": [-45, 45],
                    "rotation": [-45, 45]
                }
            })
            
            self.add_joint(f"{side}_knee", {
                "connects": [f"{side}_thigh", f"{side}_shank"],
                "dofs": 1,
                "range_of_motion": {
                    "flexion": [0, 150]
                }
            })
            
            self.add_joint(f"{side}_ankle", {
                "connects": [f"{side}_shank", f"{side}_foot"],
                "dofs": 2,
                "range_of_motion": {
                    "flexion": [-30, 50],  # Dorsiflexion/Plantarflexion
                    "inversion": [-35, 35]  # Inversion/Eversion
                }
            })
    
    def set_player_measurements(self, height: float, weight: float, limb_lengths: Dict = None) -> None:
        """Set player-specific anthropometric measurements."""
        self.anthropometrics["height"] = height
        self.anthropometrics["weight"] = weight
        
        # Calculate segment lengths based on height
        for segment_id, segment_data in self.segments.items():
            segment_length = height * segment_data["length_percentage"]
            self.segments[segment_id]["length"] = segment_length
        
        # Override with measured limb lengths if provided
        if limb_lengths:
            for segment_id, length in limb_lengths.items():
                if segment_id in self.segments:
                    self.segments[segment_id]["length"] = length
        
        # Calculate segment masses based on weight
        for segment_id, segment_data in self.segments.items():
            segment_mass = weight * segment_data["mass_percentage"]
            self.segments[segment_id]["mass"] = segment_mass
        
        logger.info(f"Set measurements for player {self.player_id}: height={height}cm, weight={weight}kg")
    
    def calculate_risk_factors(self) -> Dict:
        """Calculate injury risk factors based on the player's model."""
        # This is a placeholder for a more sophisticated risk assessment
        # In practice, this would integrate with historical injury data,
        # movement patterns, and biomechanical analysis
        
        risk_factors = {
            "general": {
                "size_vs_position": 0.0,  # Normalized score
                "muscle_imbalance": 0.0
            },
            "joints": {}
        }
        
        # Calculate size-related risk (very simplified example)
        position_weight_ranges = {
            "quarterback": [85, 110],
            "running_back": [90, 115],
            "wide_receiver": [80, 100],
            "tight_end": [100, 120],
            "offensive_lineman": [120, 160],
            "defensive_lineman": [115, 155],
            "linebacker": [100, 125],
            "defensive_back": [85, 100]
        }
        
        if self.position.lower() in position_weight_ranges:
            ideal_range = position_weight_ranges[self.position.lower()]
            weight = self.anthropometrics.get("weight", 0)
            
            if weight < ideal_range[0]:
                # Underweight for position
                weight_diff = ideal_range[0] - weight
                risk_factors["general"]["size_vs_position"] = min(1.0, weight_diff / 20)
            elif weight > ideal_range[1]:
                # Overweight for position
                weight_diff = weight - ideal_range[1]
                risk_factors["general"]["size_vs_position"] = min(1.0, weight_diff / 20)
        
        # Assess joint-specific risks
        for joint_id, joint_data in self.joints.items():
            # Placeholder for joint-specific risk analysis
            # In practice, this would consider range of motion measurements,
            # movement patterns, and historical injury data
            risk_factors["joints"][joint_id] = {
                "range_restriction": 0.0,
                "instability": 0.0
            }
        
        self.risk_factors = risk_factors
        return risk_factors
    
    def visualize_model(self, output_path: str = None) -> None:
        """Generate a simple visualization of the player model."""
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define a simple skeleton based on segment connections
        skeleton = [
            ("pelvis", "torso"),
            ("torso", "head"),
            ("torso", "left_upper_arm"),
            ("left_upper_arm", "left_forearm"),
            ("left_forearm", "left_hand"),
            ("torso", "right_upper_arm"),
            ("right_upper_arm", "right_forearm"),
            ("right_forearm", "right_hand"),
            ("pelvis", "left_thigh"),
            ("left_thigh", "left_shank"),
            ("left_shank", "left_foot"),
            ("pelvis", "right_thigh"),
            ("right_thigh", "right_shank"),
            ("right_shank", "right_foot")
        ]
        
        # Create approximate joint positions based on segment lengths
        # This is a simplified calculation - in reality would use forward kinematics
        joint_positions = {
            "pelvis": np.array([0, 0, 0]),
            "torso": np.array([0, 0, self.segments["pelvis"].get("length", 0.2)]),
            "head": np.array([0, 0, self.segments["pelvis"].get("length", 0.2) + self.segments["torso"].get("length", 0.5)])
        }
        
        # Position arms
        shoulder_width = self.anthropometrics.get("height", 180) * 0.25  # approx. shoulder width
        joint_positions["left_upper_arm"] = joint_positions["torso"] + np.array([0, shoulder_width/2, self.segments["torso"].get("length", 0.5) * 0.8])
        joint_positions["right_upper_arm"] = joint_positions["torso"] + np.array([0, -shoulder_width/2, self.segments["torso"].get("length", 0.5) * 0.8])
        
        # Left arm segments
        joint_positions["left_forearm"] = joint_positions["left_upper_arm"] + np.array([0, self.segments["left_upper_arm"].get("length", 0.3), 0])
        joint_positions["left_hand"] = joint_positions["left_forearm"] + np.array([0, self.segments["left_forearm"].get("length", 0.3), 0])
        
        # Right arm segments
        joint_positions["right_forearm"] = joint_positions["right_upper_arm"] + np.array([0, -self.segments["right_upper_arm"].get("length", 0.3), 0])
        joint_positions["right_hand"] = joint_positions["right_forearm"] + np.array([0, -self.segments["right_forearm"].get("length", 0.3), 0])
        
        # Leg positioning
        hip_width = self.anthropometrics.get("height", 180) * 0.1  # approx. hip width
        joint_positions["left_thigh"] = joint_positions["pelvis"] + np.array([0, hip_width/2, 0])
        joint_positions["right_thigh"] = joint_positions["pelvis"] + np.array([0, -hip_width/2, 0])
        
        # Left leg segments
        joint_positions["left_shank"] = joint_positions["left_thigh"] + np.array([0, 0, -self.segments["left_thigh"].get("length", 0.4)])
        joint_positions["left_foot"] = joint_positions["left_shank"] + np.array([0, 0, -self.segments["left_shank"].get("length", 0.4)])
        
        # Right leg segments
        joint_positions["right_shank"] = joint_positions["right_thigh"] + np.array([0, 0, -self.segments["right_thigh"].get("length", 0.4)])
        joint_positions["right_foot"] = joint_positions["right_shank"] + np.array([0, 0, -self.segments["right_shank"].get("length", 0.4)])
        
        # Plot the skeleton
        for connection in skeleton:
            if connection[0] in joint_positions and connection[1] in joint_positions:
                start = joint_positions[connection[0]]
                end = joint_positions[connection[1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-', linewidth=2)
        
        # Plot the joints
        for joint_name, position in joint_positions.items():
            ax.scatter(position[0], position[1], position[2], c='r', marker='o', s=50)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 2])  # Adjust the body proportions
        
        # Label the plot
        ax.set_title(f"Player Model: {self.player_id} ({self.position})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved model visualization to {output_path}")
        
        plt.close()


class MovementAnalyzer:
    """Analyzes player movements based on sensor data and biomechanical models."""
    
    def __init__(self, player_model: FootballPlayerModel = None):
        self.player_model = player_model
        logger.info(f"Initialized movement analyzer for player {player_model.player_id if player_model else 'unknown'}")
    
    def set_player_model(self, player_model: FootballPlayerModel) -> None:
        """Set the player model for analysis."""
        self.player_model = player_model
    
    def calculate_joint_angles(self, sensor_data: Dict) -> Dict:
        """
        Calculate joint angles from sensor data.
        
        Args:
            sensor_data: Processed sensor data with orientation information
            
        Returns:
            Dictionary with joint angles
        """
        if not self.player_model:
            raise ValueError("Player model not set")
        
        joint_angles = {}
        
        # This is a simplified joint angle calculation
        # In a real implementation, we would use sensor fusion and biomechanical models
        # to calculate accurate joint angles from multiple sensors
        
        # Example: Calculate knee angle from thigh and shank IMUs
        for side in ["left", "right"]:
            thigh_sensor = f"{side}_thigh_imu"
            shank_sensor = f"{side}_shank_imu"
            
            if (thigh_sensor in sensor_data and 
                shank_sensor in sensor_data and
                "orientation" in sensor_data[thigh_sensor] and
                "orientation" in sensor_data[shank_sensor]):
                
                # Get orientation in quaternions or Euler angles
                thigh_orientation = np.array([
                    sensor_data[thigh_sensor]["orientation"]["roll"],
                    sensor_data[thigh_sensor]["orientation"]["pitch"],
                    sensor_data[thigh_sensor]["orientation"]["yaw"]
                ])
                
                shank_orientation = np.array([
                    sensor_data[shank_sensor]["orientation"]["roll"],
                    sensor_data[shank_sensor]["orientation"]["pitch"],
                    sensor_data[shank_sensor]["orientation"]["yaw"]
                ])
                
                # Convert Euler angles to rotation matrices
                thigh_rot = R.from_euler('xyz', thigh_orientation)
                shank_rot = R.from_euler('xyz', shank_orientation)
                
                # Calculate relative orientation
                relative_rot = thigh_rot.inv() * shank_rot
                relative_euler = relative_rot.as_euler('xyz')
                
                # In a real implementation, we would convert this to anatomical angles
                # Here we'll use a simplified calculation
                knee_angle = abs(relative_euler[0])  # Use the first component as a simple approximation
                
                joint_angles[f"{side}_knee"] = {
                    "flexion": float(knee_angle),
                    "raw_angles": relative_euler.tolist()
                }
        
        return joint_angles
    
    def analyze_quarterback_throw(self, sensor_data_sequence: List[Dict]) -> Dict:
        """
        Analyze quarterback throwing mechanics from a sequence of sensor data.
        
        Args:
            sensor_data_sequence: List of sensor data frames covering a throwing motion
            
        Returns:
            Analysis results with metrics and assessments
        """
        if not self.player_model or self.player_model.position.lower() != "quarterback":
            raise ValueError("This analysis requires a quarterback player model")
        
        # In a real implementation, this would be a sophisticated analysis
        # of arm path, torso rotation, weight transfer, etc.
        # Here we'll implement a simplified version focusing on arm movement
        
        # Extract key sensors we need for QB analysis
        arm_sensor = "right_upper_arm_imu"  # Assuming right-handed QB
        forearm_sensor = "right_forearm_imu"
        torso_sensor = "upper_back_imu"
        
        # Initialize metrics
        max_arm_angular_velocity = 0
        max_torso_rotation_rate = 0
        arm_path_consistency = 0
        release_timing = 0
        
        # Extract time series of orientations and angular velocities
        arm_orientations = []
        arm_angular_velocities = []
        torso_orientations = []
        
        for frame in sensor_data_sequence:
            # Extract arm data
            if arm_sensor in frame["processed_sensor_data"]:
                arm_data = frame["processed_sensor_data"][arm_sensor]
                if "orientation" in arm_data and "filtered" in arm_data:
                    arm_orientations.append([
                        arm_data["orientation"]["roll"],
                        arm_data["orientation"]["pitch"],
                        arm_data["orientation"]["yaw"]
                    ])
                    arm_angular_velocities.append(arm_data["filtered"]["angular_velocity"])
            
            # Extract torso data
            if torso_sensor in frame["processed_sensor_data"]:
                torso_data = frame["processed_sensor_data"][torso_sensor]
                if "orientation" in torso_data:
                    torso_orientations.append([
                        torso_data["orientation"]["roll"],
                        torso_data["orientation"]["pitch"],
                        torso_data["orientation"]["yaw"]
                    ])
        
        # Convert to numpy arrays for analysis
        arm_orientations = np.array(arm_orientations)
        arm_angular_velocities = np.array(arm_angular_velocities)
        torso_orientations = np.array(torso_orientations)
        
        # Calculate metrics (simplified)
        if len(arm_angular_velocities) > 0:
            # Find maximum angular velocity magnitude
            arm_ang_vel_magnitude = np.linalg.norm(arm_angular_velocities, axis=1)
            max_arm_angular_velocity = float(np.max(arm_ang_vel_magnitude))
        
        if len(torso_orientations) > 1:
            # Calculate torso rotation rate between frames
            torso_yaw_diff = np.diff(torso_orientations[:, 2])
            max_torso_rotation_rate = float(np.max(np.abs(torso_yaw_diff)))
        
        if len(arm_orientations) > 0:
            # Simplified arm path consistency - standard deviation of pitch during key part of throw
            # In reality, we'd identify the throwing phase first
            middle_third = slice(len(arm_orientations) // 3, 2 * len(arm_orientations) // 3)
            arm_path_consistency = float(1.0 / (1.0 + np.std(arm_orientations[middle_third, 1])))
        
        # Compile analysis results
        analysis = {
            "metrics": {
                "max_arm_angular_velocity": max_arm_angular_velocity,
                "max_torso_rotation_rate": max_torso_rotation_rate,
                "arm_path_consistency": arm_path_consistency
            },
            "assessments": {
                "arm_speed": self._rate_metric(max_arm_angular_velocity, [5, 10, 15]),
                "torso_rotation": self._rate_metric(max_torso_rotation_rate, [0.5, 1.0, 1.5]),
                "arm_path_consistency": self._rate_metric(arm_path_consistency, [0.3, 0.6, 0.9])
            },
            "recommendations": []
        }
        
        # Generate recommendations based on metrics
        if analysis["assessments"]["arm_speed"] < 3:
            analysis["recommendations"].append("Focus on increasing arm speed through plyometric exercises")
        
        if analysis["assessments"]["torso_rotation"] < 3:
            analysis["recommendations"].append("Improve torso rotation by working on core strength and rotational flexibility")
        
        if analysis["assessments"]["arm_path_consistency"] < 3:
            analysis["recommendations"].append("Work on arm path consistency with guided throwing drills")
        
        return analysis
    
    def analyze_lineman_block(self, sensor_data_sequence: List[Dict]) -> Dict:
        """
        Analyze offensive/defensive lineman blocking technique.
        
        Args:
            sensor_data_sequence: List of sensor data frames covering a blocking motion
            
        Returns:
            Analysis results with metrics and assessments
        """
        if not self.player_model or "lineman" not in self.player_model.position.lower():
            raise ValueError("This analysis requires a lineman player model")
        
        # Initialize metrics
        leverage_score = 0
        explosion_score = 0
        hand_placement_score = 0
        balance_score = 0
        
        # Extract key sensors
        pelvis_sensor = "pelvis_imu"
        upper_back_sensor = "upper_back_imu"
        
        # Extract time series data
        pelvis_heights = []
        torso_angles = []
        acceleration_peaks = []
        
        for frame in sensor_data_sequence:
            # Extract pelvis height data (approximation for pad level)
            if pelvis_sensor in frame["processed_sensor_data"]:
                pelvis_data = frame["processed_sensor_data"][pelvis_sensor]
                if "filtered" in pelvis_data and "acceleration" in pelvis_data["filtered"]:
                    # Assume z-component is vertical
                    vertical_accel = pelvis_data["filtered"]["acceleration"][2]
                    pelvis_heights.append(vertical_accel)  # Using acceleration as proxy for movement
                    
                    # Measure acceleration magnitude for explosion
                    accel_magnitude = np.linalg.norm(pelvis_data["filtered"]["acceleration"])
                    acceleration_peaks.append(accel_magnitude)
            
            # Extract torso orientation for leverage analysis
            if upper_back_sensor in frame["processed_sensor_data"]:
                torso_data = frame["processed_sensor_data"][upper_back_sensor]
                if "orientation" in torso_data:
                    # Pitch gives forward lean
                    torso_pitch = torso_data["orientation"]["pitch"]
                    torso_angles.append(torso_pitch)
        
        # Calculate metrics (simplified)
        if pelvis_heights:
            # Lower pad level is better
            min_pelvis_height = np.min(pelvis_heights)
            leverage_score = self._normalize_metric(-min_pelvis_height, [-15, -5, 0])
        
        if acceleration_peaks:
            # Higher acceleration peak indicates better explosion
            max_acceleration = np.max(acceleration_peaks)
            explosion_score = self._normalize_metric(max_acceleration, [10, 20, 30])
        
        if torso_angles:
            # Ideal torso angle for blocking is ~45° forward
            mean_torso_angle = np.mean(torso_angles)
            forward_lean = abs(mean_torso_angle - 45)  # How close to ideal 45°
            balance_score = self._normalize_metric(-forward_lean, [-30, -15, -5])  # Negative because smaller is better
        
        # Compile analysis results
        analysis = {
            "metrics": {
                "leverage_score": float(leverage_score),
                "explosion_score": float(explosion_score),
                "balance_score": float(balance_score)
            },
            "assessments": {
                "leverage": self._rate_metric(leverage_score, [0.3, 0.6, 0.8]),
                "explosion": self._rate_metric(explosion_score, [0.3, 0.6, 0.8]),
                "balance": self._rate_metric(balance_score, [0.3, 0.6, 0.8])
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["assessments"]["leverage"] < 3:
            analysis["recommendations"].append("Work on lowering pad level with sled drills")
        
        if analysis["assessments"]["explosion"] < 3:
            analysis["recommendations"].append("Improve initial explosion with plyometric and power clean exercises")
        
        if analysis["assessments"]["balance"] < 3:
            analysis["recommendations"].append("Focus on core strength and balance drills to maintain proper blocking position")
        
        return analysis
    
    def _normalize_metric(self, value: float, ranges: List[float]) -> float:
        """Normalize a metric to a 0-1 scale based on given ranges."""
        if value <= ranges[0]:
            return 0.0
        elif value >= ranges[2]:
            return 1.0
        elif value < ranges[1]:
            # Linear interpolation between ranges[0] and ranges[1]
            return (value - ranges[0]) / (ranges[1] - ranges[0]) * 0.5
        else:
            # Linear interpolation between ranges[1] and ranges[2]
            return 0.5 + (value - ranges[1]) / (ranges[2] - ranges[1]) * 0.5
    
    def _rate_metric(self, value: float, thresholds: List[float]) -> int:
        """Rate a metric on a 1-4 scale based on thresholds."""
        if value < thresholds[0]:
            return 1
        elif value < thresholds[1]:
            return 2
        elif value < thresholds[2]:
            return 3
        else:
            return 4
    
    def identify_movement_pattern(self, sensor_data_sequence: List[Dict]) -> str:
        """
        Identify the type of football movement being performed.
        
        Args:
            sensor_data_sequence: List of sensor data frames covering a movement
            
        Returns:
            String identifying the movement type
        """
        # This is a simplified placeholder for movement recognition
        # In a real implementation, this would use ML classification
        
        # Extract features for classification
        features = self._extract_movement_features(sensor_data_sequence)
        
        # Simple rules-based classification (placeholder)
        # In reality, would use trained classifier
        if features["max_vertical_acceleration"] > 25:
            return "jump"
        elif features["max_angular_velocity"] > 10 and features["torso_rotation_range"] > 90:
            if self.player_model.position.lower() == "quarterback":
                return "throw"
            else:
                return "rotation"
        elif features["linear_acceleration_peak"] > 20:
            return "sprint"
        elif features["vertical_displacement_range"] < 0.3 and features["duration"] > 2.0:
            if "lineman" in self.player_model.position.lower():
                return "block"
            else:
                return "stance"
        else:
            return "unknown"
    
    def _extract_movement_features(self, sensor_data_sequence: List[Dict]) -> Dict:
        """Extract key features from a movement sequence for classification."""
        features = {
            "duration": 0,
            "max_vertical_acceleration": 0,
            "max_angular_velocity": 0,
            "linear_acceleration_peak": 0,
            "torso_rotation_range": 0,
            "vertical_displacement_range": 0
        }
        
        # Extract time information
        if len(sensor_data_sequence) >= 2:
            start_time = sensor_data_sequence[0]["timestamp"]
            end_time = sensor_data_sequence[-1]["timestamp"]
            features["duration"] = end_time - start_time
        
        # Process sensor data
        pelvis_vertical_accels = []
        torso_rotations = []
        max_ang_vel = 0
        max_lin_accel = 0
        
        for frame in sensor_data_sequence:
            # Extract pelvis data for vertical movement
            if "pelvis_imu" in frame["processed_sensor_data"]:
                pelvis_data = frame["processed_sensor_data"]["pelvis_imu"]
                if "filtered" in pelvis_data and "acceleration" in pelvis_data["filtered"]:
                    vert_accel = pelvis_data["filtered"]["acceleration"][2]  # Z-axis vertical
                    pelvis_vertical_accels.append(vert_accel)
                    
                    # Track peak acceleration
                    accel_mag = np.linalg.norm(pelvis_data["filtered"]["acceleration"])
                    max_lin_accel = max(max_lin_accel, accel_mag)
            
            # Extract torso data for rotation
            if "upper_back_imu" in frame["processed_sensor_data"]:
                torso_data = frame["processed_sensor_data"]["upper_back_imu"]
                if "orientation" in torso_data:
                    torso_rotations.append(torso_data["orientation"]["yaw"])
                
                if "filtered" in torso_data and "angular_velocity" in torso_data["filtered"]:
                    ang_vel_mag = np.linalg.norm(torso_data["filtered"]["angular_velocity"])
                    max_ang_vel = max(max_ang_vel, ang_vel_mag)
        
        # Calculate final features
        if pelvis_vertical_accels:
            features["max_vertical_acceleration"] = max(abs(a) for a in pelvis_vertical_accels)
            
            # Calculate vertical displacement (simplified)
            # In reality would do double integration of acceleration
            features["vertical_displacement_range"] = max(pelvis_vertical_accels) - min(pelvis_vertical_accels)
        
        if torso_rotations:
            features["torso_rotation_range"] = max(torso_rotations) - min(torso_rotations)
        
        features["max_angular_velocity"] = max_ang_vel
        features["linear_acceleration_peak"] = max_lin_accel
        
        return features


class TechniqueAnalyzer:
    """Analyzes and scores specific football techniques based on biomechanical models."""
    
    def __init__(self, model_path: str = None):
        self.technique_models = {}
        self.movement_analyzer = MovementAnalyzer()
        
        # Load technique models if provided
        if model_path and os.path.exists(model_path):
            self._load_technique_models(model_path)
        
        logger.info("Initialized technique analyzer")
    
    def _load_technique_models(self, model_path: str) -> None:
        """Load technique models from file."""
        # In a real implementation, this would load trained ML models
        # or reference data for technique comparison
        pass
    
    def add_player_model(self, player_model: FootballPlayerModel) -> None:
        """Add a player model to the technique analyzer."""
        self.movement_analyzer.set_player_model(player_model)
    
    def analyze_technique(self, technique_name: str, sensor_data_sequence: List[Dict]) -> Dict:
        """
        Analyze a specific football technique.
        
        Args:
            technique_name: Name of the technique to analyze
            sensor_data_sequence: List of sensor data frames covering the technique
            
        Returns:
            Analysis results with scores and recommendations
        """
        # Identify the right analyzer based on technique name
        if technique_name == "qb_throw":
            return self.movement_analyzer.analyze_quarterback_throw(sensor_data_sequence)
        elif technique_name == "lineman_block":
            return self.movement_analyzer.analyze_lineman_block(sensor_data_sequence)
        elif technique_name == "sprint":
            return self._analyze_sprint(sensor_data_sequence)
        elif technique_name == "tackle":
            return self._analyze_tackle(sensor_data_sequence)
        elif technique_name == "route_running":
            return self._analyze_route_running(sensor_data_sequence)
        else:
            raise ValueError(f"Unknown technique: {technique_name}")
    
    def _analyze_sprint(self, sensor_data_sequence: List[Dict]) -> Dict:
        """Analyze sprinting technique."""
        # Placeholder for sprint analysis
        return {"status": "not implemented"}
    
    def _analyze_tackle(self, sensor_data_sequence: List[Dict]) -> Dict:
        """Analyze tackling technique with focus on safety."""
        # Placeholder for tackle analysis
        return {"status": "not implemented"}
    
    def _analyze_route_running(self, sensor_data_sequence: List[Dict]) -> Dict:
        """Analyze route running technique for receivers."""
        # Placeholder for route running analysis
        return {"status": "not implemented"}


class PlayerPerformanceProfile:
    """Maintains and updates a player's performance profile based on biomechanical analysis."""
    
    def __init__(self, player_id: str, position: str):
        self.player_id = player_id
        self.position = position
        self.biomechanical_model = FootballPlayerModel(player_id, position)
        self.technique_scores = {}
        self.movement_efficiency = {}
        self.injury_risk = {}
        self.performance_trends = {}
        self.last_updated = datetime.now()
        
        logger.info(f"Initialized performance profile for {player_id} ({position})")
    
    def update_anthropometrics(self, height: float, weight: float, limb_lengths: Dict = None) -> None:
        """Update player anthropometric measurements."""
        self.biomechanical_model.set_player_measurements(height, weight, limb_lengths)
        self.last_updated = datetime.now()
    
    def update_technique_score(self, technique: str, score: float, timestamp: float = None) -> None:
        """Update a technique score with a new measurement."""
        if technique not in self.technique_scores:
            self.technique_scores[technique] = []
        
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        self.technique_scores[technique].append({
            "timestamp": timestamp,
            "score": score
        })
        
        self.last_updated = datetime.now()
    
    def update_injury_risk(self, risk_assessment: Dict) -> None:
        """Update injury risk assessment."""
        self.injury_risk = risk_assessment
        self.injury_risk["timestamp"] = datetime.now().timestamp()
        self.last_updated = datetime.now()
    
    def get_technique_trend(self, technique: str, days: int = 30) -> Dict:
        """Get trend for a specific technique over time."""
        if technique not in self.technique_scores:
            return {
                "technique": technique,
                "available": False,
                "message": "No data available for this technique"
            }
        
        # Filter to requested time period
        cutoff_time = (datetime.now() - pd.Timedelta(days=days)).timestamp()
        recent_scores = [s for s in self.technique_scores[technique] if s["timestamp"] >= cutoff_time]
        
        if len(recent_scores) < 2:
            return {
                "technique": technique,
                "available": False,
                "message": f"Insufficient data in the last {days} days"
            }
        
        # Calculate trend
        scores = [s["score"] for s in recent_scores]
        timestamps = [s["timestamp"] for s in recent_scores]
        
        # Calculate linear regression
        slope, intercept = np.polyfit(timestamps, scores, 1)
        
        # Calculate additional statistics
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        return {
            "technique": technique,
            "available": True,
            "data_points": len(recent_scores),
            "mean_score": float(mean_score),
            "max_score": float(max_score),
            "min_score": float(min_score),
            "trend_slope": float(slope),  # positive means improving
            "trend_direction": "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable"
        }
    
    def generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        report = {
            "player_id": self.player_id,
            "position": self.position,
            "report_date": datetime.now().isoformat(),
            "anthropometrics": self.biomechanical_model.anthropometrics,
            "technique_summary": {},
            "injury_risk_summary": {},
            "recommendations": []
        }
        
        # Summarize technique scores
        for technique, scores in self.technique_scores.items():
            if scores:
                recent_scores = sorted(scores, key=lambda x: x["timestamp"], reverse=True)
                latest_score = recent_scores[0]["score"]
                trend = self.get_technique_trend(technique)
                
                report["technique_summary"][technique] = {
                    "latest_score": latest_score,
                    "trend": trend.get("trend_direction", "unknown")
                }
                
                # Generate recommendation if technique needs improvement
                if latest_score < 0.7 and trend.get("available", False):
                    report["recommendations"].append({
                        "area": technique,
                        "recommendation": f"Focus on improving {technique.replace('_', ' ')} technique",
                        "priority": "high" if latest_score < 0.5 else "medium"
                    })
        
        # Summarize injury risk
        if self.injury_risk:
            report["injury_risk_summary"] = {
                "overall_risk": self.injury_risk.get("general", {}).get("size_vs_position", 0),
                "high_risk_areas": []
            }
            
            # Identify high risk areas
            for joint, risks in self.injury_risk.get("joints", {}).items():
                for risk_type, risk_value in risks.items():
                    if risk_value > 0.7:
                        report["injury_risk_summary"]["high_risk_areas"].append({
                            "joint": joint,
                            "risk_type": risk_type,
                            "risk_value": risk_value
                        })
                        
                        # Generate recommendation for high risk
                        report["recommendations"].append({
                            "area": f"{joint} {risk_type}",
                            "recommendation": f"Address {risk_type} issues in the {joint} joint",
                            "priority": "high" if risk_value > 0.8 else "medium"
                        })
        
        return report
    
    def save_profile(self, output_path: str) -> str:
        """Save the player performance profile to file."""
        os.makedirs(output_path, exist_ok=True)
        
        profile_data = {
            "player_id": self.player_id,
            "position": self.position,
            "last_updated": self.last_updated.isoformat(),
            "technique_scores": self.technique_scores,
            "injury_risk": self.injury_risk,
            "performance_trends": self.performance_trends
        }
        
        filename = os.path.join(output_path, f"profile_{self.player_id}.json")
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Saved performance profile to {filename}")
        
        # Also save the biomechanical model
        model_filename = self.biomechanical_model.save_model(output_path)
        
        return filename
    
    @classmethod
    def load_profile(cls, profile_path: str, model_path: str = None):
        """Load a player performance profile from file."""
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        player_id = profile_data["player_id"]
        position = profile_data["position"]
        
        profile = cls(player_id, position)
        profile.technique_scores = profile_data["technique_scores"]
        profile.injury_risk = profile_data["injury_risk"]
        profile.performance_trends = profile_data.get("performance_trends", {})
        profile.last_updated = datetime.fromisoformat(profile_data["last_updated"])
        
        # Load biomechanical model if path provided
        if model_path and os.path.exists(model_path):
            profile.biomechanical_model = FootballPlayerModel.load_model(model_path)
        
        logger.info(f"Loaded performance profile for {player_id} from {profile_path}")
        return profile
