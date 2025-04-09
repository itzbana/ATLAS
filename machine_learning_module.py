"""
ATLAS Football - Machine Learning Module

This module handles training and inference for various machine learning models
that power the ATLAS system's analysis and prediction capabilities.
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atlas_ml")


class FeatureExtractor:
    """Extracts features from processed sensor data for machine learning models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_sets = {
            "movement_classification": self._extract_movement_classification_features,
            "technique_quality": self._extract_technique_quality_features,
            "injury_risk": self._extract_injury_risk_features,
            "performance_prediction": self._extract_performance_prediction_features
        }
        logger.info("Initialized feature extractor")
    
    def extract_features(self, data: Union[Dict, List[Dict]], feature_set: str) -> pd.DataFrame:
        """
        Extract features from sensor data based on the requested feature set.
        
        Args:
            data: Processed sensor data (single data point or sequence)
            feature_set: Name of the feature set to extract
            
        Returns:
            DataFrame with extracted features
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        # Call the appropriate feature extraction function
        return self.feature_sets[feature_set](data)
    
    def _extract_movement_classification_features(self, data_sequence: List[Dict]) -> pd.DataFrame:
        """
        Extract features for movement classification.
        
        Args:
            data_sequence: List of sensor data frames covering a movement
            
        Returns:
            DataFrame with extracted features
        """
        # For movement classification, we need to process a sequence of frames
        if not isinstance(data_sequence, list):
            raise ValueError("Movement classification requires a sequence of data frames")
        
        # Initialize features dictionary
        features = {
            # Temporal features
            "sequence_duration": 0,
            "frame_count": len(data_sequence),
            
            # Acceleration features
            "max_accel_x": 0, "max_accel_y": 0, "max_accel_z": 0,
            "min_accel_x": 0, "min_accel_y": 0, "min_accel_z": 0,
            "mean_accel_x": 0, "mean_accel_y": 0, "mean_accel_z": 0,
            "std_accel_x": 0, "std_accel_y": 0, "std_accel_z": 0,
            "peak_accel_magnitude": 0,
            
            # Angular velocity features
            "max_gyro_x": 0, "max_gyro_y": 0, "max_gyro_z": 0,
            "min_gyro_x": 0, "min_gyro_y": 0, "min_gyro_z": 0,
            "mean_gyro_x": 0, "mean_gyro_y": 0, "mean_gyro_z": 0,
            "std_gyro_x": 0, "std_gyro_y": 0, "std_gyro_z": 0,
            "peak_gyro_magnitude": 0,
            
            # Orientation features
            "max_roll": 0, "max_pitch": 0, "max_yaw": 0,
            "min_roll": 0, "min_pitch": 0, "min_yaw": 0,
            "range_roll": 0, "range_pitch": 0, "range_yaw": 0,
            
            # Energy and frequency features
            "total_energy": 0,
            "dominant_frequency": 0,
            
            # Positional sensors (if available)
            "displacement_x": 0, "displacement_y": 0, "displacement_z": 0,
            "max_velocity": 0
        }
        
        # Extract key sensors data
        accel_data = []
        gyro_data = []
        orientation_data = []
        timestamps = []
        
        primary_sensors = ["pelvis_imu", "upper_back_imu", "helmet_imu"]
        for frame in data_sequence:
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                # Use primary sensors only
                sensor_name = "_".join(sensor_key.split("_")[:-1])  # Extract base name without ID
                if sensor_name in primary_sensors:
                    # Extract acceleration data
                    if "filtered" in sensor_data and "acceleration" in sensor_data["filtered"]:
                        accel_data.append(sensor_data["filtered"]["acceleration"])
                    
                    # Extract angular velocity data
                    if "filtered" in sensor_data and "angular_velocity" in sensor_data["filtered"]:
                        gyro_data.append(sensor_data["filtered"]["angular_velocity"])
                    
                    # Extract orientation data
                    if "orientation" in sensor_data:
                        orientation = [
                            sensor_data["orientation"]["roll"],
                            sensor_data["orientation"]["pitch"],
                            sensor_data["orientation"]["yaw"]
                        ]
                        orientation_data.append(orientation)
            
            timestamps.append(frame.get("timestamp", 0))
        
        # Convert to numpy arrays for easier processing
        if accel_data:
            accel_data = np.array(accel_data)
            features["max_accel_x"] = float(np.max(accel_data[:, 0]))
            features["max_accel_y"] = float(np.max(accel_data[:, 1]))
            features["max_accel_z"] = float(np.max(accel_data[:, 2]))
            features["min_accel_x"] = float(np.min(accel_data[:, 0]))
            features["min_accel_y"] = float(np.min(accel_data[:, 1]))
            features["min_accel_z"] = float(np.min(accel_data[:, 2]))
            features["mean_accel_x"] = float(np.mean(accel_data[:, 0]))
            features["mean_accel_y"] = float(np.mean(accel_data[:, 1]))
            features["mean_accel_z"] = float(np.mean(accel_data[:, 2]))
            features["std_accel_x"] = float(np.std(accel_data[:, 0]))
            features["std_accel_y"] = float(np.std(accel_data[:, 1]))
            features["std_accel_z"] = float(np.std(accel_data[:, 2]))
            
            # Calculate magnitude
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            features["peak_accel_magnitude"] = float(np.max(accel_magnitude))
            
            # Energy calculation (simplified)
            features["total_energy"] = float(np.sum(np.square(accel_magnitude)))
        
        if gyro_data:
            gyro_data = np.array(gyro_data)
            features["max_gyro_x"] = float(np.max(gyro_data[:, 0]))
            features["max_gyro_y"] = float(np.max(gyro_data[:, 1]))
            features["max_gyro_z"] = float(np.max(gyro_data[:, 2]))
            features["min_gyro_x"] = float(np.min(gyro_data[:, 0]))
            features["min_gyro_y"] = float(np.min(gyro_data[:, 1]))
            features["min_gyro_z"] = float(np.min(gyro_data[:, 2]))
            features["mean_gyro_x"] = float(np.mean(gyro_data[:, 0]))
            features["mean_gyro_y"] = float(np.mean(gyro_data[:, 1]))
            features["mean_gyro_z"] = float(np.mean(gyro_data[:, 2]))
            features["std_gyro_x"] = float(np.std(gyro_data[:, 0]))
            features["std_gyro_y"] = float(np.std(gyro_data[:, 1]))
            features["std_gyro_z"] = float(np.std(gyro_data[:, 2]))
            
            # Calculate magnitude
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            features["peak_gyro_magnitude"] = float(np.max(gyro_magnitude))
        
        if orientation_data:
            orientation_data = np.array(orientation_data)
            features["max_roll"] = float(np.max(orientation_data[:, 0]))
            features["max_pitch"] = float(np.max(orientation_data[:, 1]))
            features["max_yaw"] = float(np.max(orientation_data[:, 2]))
            features["min_roll"] = float(np.min(orientation_data[:, 0]))
            features["min_pitch"] = float(np.min(orientation_data[:, 1]))
            features["min_yaw"] = float(np.min(orientation_data[:, 2]))
            features["range_roll"] = features["max_roll"] - features["min_roll"]
            features["range_pitch"] = features["max_pitch"] - features["min_pitch"]
            features["range_yaw"] = features["max_yaw"] - features["min_yaw"]
        
        if len(timestamps) >= 2:
            features["sequence_duration"] = timestamps[-1] - timestamps[0]
        
        # Convert to DataFrame
        return pd.DataFrame([features])
    
    def _extract_technique_quality_features(self, data_sequence: List[Dict]) -> pd.DataFrame:
        """
        Extract features for technique quality assessment.
        
        Args:
            data_sequence: List of sensor data frames covering a technique
            
        Returns:
            DataFrame with extracted features
        """
        # Start with basic movement features
        base_features = self._extract_movement_classification_features(data_sequence)
        
        # Add technique-specific features
        # This would vary based on the technique being analyzed
        # For demonstration, we'll add some generic technique features
        technique_features = {
            # Smoothness metrics
            "jerk_cost": 0,  # Measure of movement smoothness
            "movement_symmetry": 0,  # Left/right symmetry
            
            # Timing metrics
            "phase_durations": [],  # Duration of movement phases
            "relative_timing": [],  # Timing of key events
            
            # Technique-specific metrics
            "form_consistency": 0,  # Consistency of form
            "power_efficiency": 0,  # Efficiency of power transfer
            "stability_index": 0    # Measure of stability
        }
        
        # Calculate jerk cost (derivative of acceleration)
        accel_sequences = []
        for frame in data_sequence:
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                if "derived" in sensor_data and "jerk" in sensor_data["derived"]:
                    accel_sequences.append(sensor_data["derived"]["jerk"])
        
        if accel_sequences:
            jerk_data = np.array(accel_sequences)
            jerk_magnitude = np.linalg.norm(jerk_data, axis=1)
            technique_features["jerk_cost"] = float(np.mean(np.square(jerk_magnitude)))
        
        # Calculate movement symmetry
        left_data = []
        right_data = []
        for frame in data_sequence:
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                if "left" in sensor_key and "filtered" in sensor_data:
                    left_data.append(sensor_data["filtered"]["acceleration"])
                elif "right" in sensor_key and "filtered" in sensor_data:
                    right_data.append(sensor_data["filtered"]["acceleration"])
        
        if left_data and right_data:
            left_data = np.array(left_data)
            right_data = np.array(right_data)
            
            # Simple symmetry calculation
            left_energy = np.sum(np.square(np.linalg.norm(left_data, axis=1)))
            right_energy = np.sum(np.square(np.linalg.norm(right_data, axis=1)))
            
            # If perfectly symmetric, will equal 1.0
            if left_energy > 0 and right_energy > 0:
                technique_features["movement_symmetry"] = float(min(left_energy, right_energy) / max(left_energy, right_energy))
        
        # Add technique features to base features
        for key, value in technique_features.items():
            if not isinstance(value, list):
                base_features[key] = value
        
        return base_features
    
    def _extract_injury_risk_features(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Extract features for injury risk assessment.
        
        Args:
            data: Processed sensor data (single data point or sequence)
            
        Returns:
            DataFrame with extracted features
        """
        if isinstance(data, list):
            # For sequence data, calculate different features
            return self._extract_injury_risk_from_sequence(data)
        else:
            # For individual data points
            return self._extract_injury_risk_from_point(data)
    
    def _extract_injury_risk_from_sequence(self, data_sequence: List[Dict]) -> pd.DataFrame:
        """Extract injury risk features from a movement sequence."""
        # Get base movement features
        base_features = self._extract_movement_classification_features(data_sequence)
        
        # Add injury risk specific features
        risk_features = {
            # Impact metrics
            "max_impact_acceleration": 0,
            "cumulative_impact_load": 0,
            
            # Joint metrics
            "joint_torque_max": 0,
            "joint_range_of_motion_exceeded": False,
            "asymmetry_index": 0,
            
            # Movement quality metrics
            "movement_control": 0,
            "fatigue_index": 0
        }
        
        # Calculate impact metrics
        accelerations = []
        for frame in data_sequence:
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                if "filtered" in sensor_data and "acceleration" in sensor_data["filtered"]:
                    accel = sensor_data["filtered"]["acceleration"]
                    accel_mag = np.linalg.norm(accel)
                    accelerations.append(accel_mag)
        
        if accelerations:
            risk_features["max_impact_acceleration"] = float(np.max(accelerations))
            # Cumulative load is a measure of total impact exposure
            risk_features["cumulative_impact_load"] = float(np.sum(accelerations))
        
        # Check for exceeded range of motion
        joint_angles = []
        for frame in data_sequence:
            # In a real implementation, would extract calculated joint angles
            # This is a placeholder for demonstration
            pass
            
        # Calculate asymmetry index
        left_data = []
        right_data = []
        for frame in data_sequence:
            for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                if "left" in sensor_key and "filtered" in sensor_data:
                    left_data.append(sensor_data["filtered"]["acceleration"])
                elif "right" in sensor_key and "filtered" in sensor_data:
                    right_data.append(sensor_data["filtered"]["acceleration"])
        
        if left_data and right_data:
            left_data = np.array(left_data)
            right_data = np.array(right_data)
            
            # Calculate asymmetry (more sophisticated than the technique version)
            left_mean = np.mean(np.linalg.norm(left_data, axis=1))
            right_mean = np.mean(np.linalg.norm(right_data, axis=1))
            
            if left_mean > 0 or right_mean > a:
                asymmetry = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)
                risk_features["asymmetry_index"] = float(asymmetry)
        
        # Estimate fatigue (simplified)
        if len(data_sequence) > 10:
            # Compare first half to second half of sequence
            half_idx = len(data_sequence) // 2
            first_half = data_sequence[:half_idx]
            second_half = data_sequence[half_idx:]
            
            # Extract features from each half
            first_half_features = self._extract_movement_classification_features(first_half)
            second_half_features = self._extract_movement_classification_features(second_half)
            
            # Calculate fatigue index based on energy reduction
            if "total_energy" in first_half_features and "total_energy" in second_half_features:
                first_energy = first_half_features["total_energy"].values[0]
                second_energy = second_half_features["total_energy"].values[0]
                
                if first_energy > 0:
                    energy_ratio = second_energy / first_energy
                    # Lower energy in second half indicates fatigue
                    risk_features["fatigue_index"] = float(max(0, 1 - energy_ratio))
        
        # Add injury risk features to base features
        for key, value in risk_features.items():
            base_features[key] = value
        
        return base_features
    
    def _extract_injury_risk_from_point(self, data_point: Dict) -> pd.DataFrame:
        """Extract injury risk features from a single data point."""
        # This would be used for monitoring during activity
        features = {
            "timestamp": data_point.get("timestamp", 0),
            "player_id": data_point.get("player_id", "unknown"),
            "max_acceleration": 0,
            "max_angular_velocity": 0,
            "joint_loads": {},
            "asymmetry_detected": False
        }
        
        # Extract sensor data
        max_accel = 0
        max_ang_vel = 0
        
        for sensor_key, sensor_data in data_point.get("processed_sensor_data", {}).items():
            if "filtered" in sensor_data:
                if "acceleration" in sensor_data["filtered"]:
                    accel_mag = np.linalg.norm(sensor_data["filtered"]["acceleration"])
                    max_accel = max(max_accel, accel_mag)
                
                if "angular_velocity" in sensor_data["filtered"]:
                    ang_vel_mag = np.linalg.norm(sensor_data["filtered"]["angular_velocity"])
                    max_ang_vel = max(max_ang_vel, ang_vel_mag)
        
        features["max_acceleration"] = float(max_accel)
        features["max_angular_velocity"] = float(max_ang_vel)
        
        return pd.DataFrame([features])
    
    def _extract_performance_prediction_features(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Extract features for performance prediction.
        
        Args:
            data: Processed sensor data or player performance history
            
        Returns:
            DataFrame with extracted features
        """
        # For performance prediction, we typically use a mix of:
        # 1. Biomechanical efficiency metrics
        # 2. Historical performance trends
        # 3. Fatigue and readiness indicators
        
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Sequence of sensor data
            features = self._extract_movement_classification_features(data)
            
            # Add performance-specific features
            performance_features = {
                "power_output": 0,
                "movement_efficiency": 0,
                "technique_consistency": 0,
                "recovery_capacity": 0
            }
            
            # Calculate power output (simplified)
            accelerations = []
            for frame in data:
                for sensor_key, sensor_data in frame.get("processed_sensor_data", {}).items():
                    if "filtered" in sensor_data and "acceleration" in sensor_data["filtered"]:
                        accel = sensor_data["filtered"]["acceleration"]
                        accel_mag = np.linalg.norm(accel)
                        accelerations.append(accel_mag)
            
            if accelerations:
                # Simple power approximation
                performance_features["power_output"] = float(np.mean(np.array(accelerations) ** 2))
            
            # Add performance features
            for key, value in performance_features.items():
                features[key] = value
            
            return features
        
        else:
            # Performance history data
            # This would typically be a preprocessed performance record
            # For now, return empty frame as placeholder
            return pd.DataFrame()


class ModelTrainer:
    """Trains and evaluates machine learning models for different ATLAS components."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        logger.info(f"Initialized model trainer with model directory: {model_dir}")
    
    def train_movement_classifier(self, training_data: List[Dict], labels: List[str]) -> Dict:
        """
        Train a model to classify movement types.
        
        Args:
            training_data: List of sensor data sequences
            labels: Movement type labels for each sequence
            
        Returns:
            Training results with metrics
        """
        logger.info(f"Training movement classifier with {len(training_data)} samples")
        
        # Extract features for each sequence
        features = []
        for sequence in training_data:
            sequence_features = self.feature_extractor.extract_features(
                sequence, "movement_classification")
            features.append(sequence_features)
        
        # Combine all features
        X = pd.concat(features, ignore_index=True)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1": float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, "movement_classifier.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"Movement classifier trained with accuracy: {metrics['accuracy']:.4f}")
        
        return {
            "model_path": model_path,
            "metrics": metrics,
            "feature_importance": self._get_feature_importance(model, X.columns)
        }
    
    def train_technique_quality_model(self, training_data: List[Dict], 
                                     quality_scores: List[float],
                                     technique_type: str) -> Dict:
        """
        Train a model to assess technique quality.
        
        Args:
            training_data: List of sensor data sequences
            quality_scores: Expert-assigned quality scores for each sequence
            technique_type: Type of technique (e.g., "qb_throw", "tackle")
            
        Returns:
            Training results with metrics
        """
        logger.info(f"Training {technique_type} quality model with {len(training_data)} samples")
        
        # Extract features for each sequence
        features = []
        for sequence in training_data:
            sequence_features = self.feature_extractor.extract_features(
                sequence, "technique_quality")
            features.append(sequence_features)
        
        # Combine all features
        X = pd.concat(features, ignore_index=True)
        y = np.array(quality_scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(model.score(X_test, y_test))
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{technique_type}_quality_model.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"{technique_type.capitalize()} quality model trained with RMSE: {metrics['rmse']:.4f}")
        
        return {
            "model_path": model_path,
            "metrics": metrics,
            "feature_importance": self._get_feature_importance(model, X.columns)
        }
    
    def train_injury_risk_model(self, training_data: List[Dict], 
                               risk_labels: List[float]) -> Dict:
        """
        Train a model to assess injury risk.
        
        Args:
            training_data: List of sensor data sequences
            risk_labels: Injury risk scores (0-1) for each sequence
            
        Returns:
            Training results with metrics
        """
        logger.info(f"Training injury risk model with {len(training_data)} samples")
        
        # Extract features for each sequence
        features = []
        for sequence in training_data:
            sequence_features = self.feature_extractor.extract_features(
                sequence, "injury_risk")
            features.append(sequence_features)
        
        # Combine all features
        X = pd.concat(features, ignore_index=True)
        y = np.array(risk_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(model.score(X_test, y_test))
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, "injury_risk_model.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"Injury risk model trained with RMSE: {metrics['rmse']:.4f}")
        
        return {
            "model_path": model_path,
            "metrics": metrics,
            "feature_importance": self._get_feature_importance(model, X.columns)
        }
    
    def train_performance_prediction_model(self, training_data: List[Dict], 
                                          performance_metrics: List[float],
                                          metric_name: str) -> Dict:
        """
        Train a model to predict performance metrics.
        
        Args:
            training_data: List of sensor data sequences and player history
            performance_metrics: Actual performance outcomes
            metric_name: Name of the performance metric being predicted
            
        Returns:
            Training results with metrics
        """
        logger.info(f"Training {metric_name} prediction model with {len(training_data)} samples")
        
        # Extract features for each data point
        features = []
        for data in training_data:
            data_features = self.feature_extractor.extract_features(
                data, "performance_prediction")
            features.append(data_features)
        
        # Combine all features
        X = pd.concat(features, ignore_index=True)
        y = np.array(performance_metrics)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(model.score(X_test, y_test))
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{metric_name}_prediction_model.pkl")
        joblib.dump(model, model_path)
        
        logger.info(f"{metric_name} prediction model trained with RMSE: {metrics['rmse']:.4f}")
        
        return {
            "model_path": model_path,
            "metrics": metrics,
            "feature_importance": self._get_feature_importance(model, X.columns)
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from the model."""
        # Get the actual model from the pipeline
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            estimator = model.named_steps['classifier']
        elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            estimator = model.named_steps['regressor']
        else:
            estimator = model
        
        # Extract feature importance if available
        if hasattr(estimator, 'feature_importances_'):
            importance = estimator.feature_importances_
            return dict(zip(feature_names, importance))
        else:
            return {}


class ModelPredictor:
    """Makes predictions using trained machine learning models."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        logger.info(f"Initialized model predictor with model directory: {model_dir}")
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Success status
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        
        try:
            self.models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def classify_movement(self, sensor_data_sequence: List[Dict]) -> Dict:
        """
        Classify the type of movement from sensor data.
        
        Args:
            sensor_data_sequence: List of sensor data frames
            
        Returns:
            Classification results with probabilities
        """
        model_name = "movement_classifier"
        if model_name not in self.models:
            if not self.load_model(model_name):
                return {"error": "Model not available"}
        
        # Extract features
        features = self.feature_extractor.extract_features(
            sensor_data_sequence, "movement_classification")
        
        # Make prediction
        model = self.models[model_name]
        movement_class = model.predict(features)[0]
        
        result = {
            "movement_type": movement_class,
            "confidence": 0.0
        }
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            class_indices = model.classes_
            class_probs = dict(zip(class_indices, probabilities))
            result["confidence"] = float(class_probs[movement_class])
            result["all_probabilities"] = {str(k): float(v) for k, v in class_probs.items()}
        
        return result
    
    def assess_technique_quality(self, sensor_data_sequence: List[Dict], 
                                technique_type: str) -> Dict:
        """
        Assess the quality of a technique execution.
        
        Args:
            sensor_data_sequence: List of sensor data frames
            technique_type: Type of technique to assess
            
        Returns:
            Quality assessment with score and confidence
        """
        model_name = f"{technique_type}_quality_model"
        if model_name not in self.models:
            if not self.load_model(model_name):
                return {"error": "Model not available"}
        
        # Extract features
        features = self.feature_extractor.extract_features(
            sensor_data_sequence, "technique_quality")
        
        # Make prediction
        model = self.models[model_name]
        quality_score = float(model.predict(features)[0])
        
        # Ensure score is in 0-1 range
        quality_score = max(0, min(1, quality_score))
        
        return {
            "technique_type": technique_type,
            "quality_score": quality_score,
            "rating": self._score_to_rating(quality_score)
        }
    
    def assess_injury_risk(self, sensor_data_sequence: List[Dict]) -> Dict:
        """
        Assess injury risk from sensor data.
        
        Args:
            sensor_data_sequence: List of sensor data frames
            
        Returns:
            Risk assessment with score and factors
        """
        model_name = "injury_risk_model"
        if model_name not in self.models:
            if not self.load_model(model_name):
                return {"error": "Model not available"}
        
        # Extract features
        features = self.feature_extractor.extract_features(
            sensor_data_sequence, "injury_risk")
        
        # Make prediction
        model = self.models[model_name]
        risk_score = float(model.predict(features)[0])
        
        # Ensure score is in 0-1 range
        risk_score = max(0, min(1, risk_score))
        
        # Identify key risk factors
        risk_factors = self._identify_risk_factors(features)
        
        return {
            "risk_score": risk_score,
            "risk_level": self._risk_score_to_level(risk_score),
            "risk_factors": risk_factors
        }
    
    def predict_performance(self, data: Union[Dict, List[Dict]], 
                           metric_name: str) -> Dict:
        """
        Predict performance metrics.
        
        Args:
            data: Sensor data and/or player history
            metric_name: Performance metric to predict
            
        Returns:
            Performance prediction with confidence interval
        """
        model_name = f"{metric_name}_prediction_model"
        if model_name not in self.models:
            if not self.load_model(model_name):
                return {"error": "Model not available"}
        
        # Extract features
        features = self.feature_extractor.extract_features(
            data, "performance_prediction")
        
        # Make prediction
        model = self.models[model_name]
        predicted_value = float(model.predict(features)[0])
        
        return {
            "metric_name": metric_name,
            "predicted_value": predicted_value,
            # In a real implementation, we would calculate a proper confidence interval
            "confidence_interval": [predicted_value * 0.9, predicted_value * 1.1]
        }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert a numeric score to a rating category."""
        if score < 0.3:
            return "Poor"
        elif score < 0.5:
            return "Fair"
        elif score < 0.7:
            return "Good"
        elif score < 0.9:
            return "Excellent"
        else:
            return "Elite"
    
    def _risk_score_to_level(self, score: float) -> str:
        """Convert a risk score to a risk level category."""
        if score < 0.2:
            return "Very Low"
        elif score < 0.4:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _identify_risk_factors(self, features: pd.DataFrame) -> List[Dict]:
        """
        Identify key risk factors from feature values.
        
        This is a simplified implementation - in practice would be more sophisticated.
        """
        risk_factors = []
        
        # Examine key features for risk indications
        feature_thresholds = {
            "max_impact_acceleration": 20,
            "asymmetry_index": 0.25,
            "fatigue_index": 0.7,
        }
        
        feature_descriptions = {
            "max_impact_acceleration": "High impact forces",
            "asymmetry_index": "Movement asymmetry",
            "fatigue_index": "Fatigue indicators",
        }
        
        for feature, threshold in feature_thresholds.items():
            if feature in features and features[feature].values[0] > threshold:
                risk_factors.append({
                    "factor": feature_descriptions.get(feature, feature),
                    "severity": float(features[feature].values[0]),
                    "description": f"Elevated {feature_descriptions.get(feature, feature)}"
                })
        
        return risk_factors
