"""
ATLAS Football - Main Application

This module integrates all components of the ATLAS system and provides
a complete workflow for football player analysis.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import argparse
from pathlib import Path

# Import ATLAS components
from sensor_data_collection import DataCollectionSession, PlayerSensorSystem, IMUSensor
from data_processing_pipeline import DataPipeline, PlayerDataProcessor
from biomechanical_modeling import FootballPlayerModel, TechniqueAnalyzer, PlayerPerformanceProfile
from machine_learning_module import FeatureExtractor, ModelTrainer, ModelPredictor
from visualization_interface import VisualizationGenerator, DashboardGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("atlas.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("atlas_main")


class ATLASSystem:
    """Main class for the ATLAS Football System."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the ATLAS system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Create directories
        self.data_dir = self.config.get("data_directory", "data")
        self.model_dir = self.config.get("model_directory", "models")
        self.output_dir = self.config.get("output_directory", "output")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.data_pipeline = DataPipeline(config_path)
        self.model_trainer = ModelTrainer(self.model_dir)
        self.model_predictor = ModelPredictor(self.model_dir)
        self.viz_generator = VisualizationGenerator(os.path.join(self.output_dir, "visualizations"))
        self.dashboard_generator = DashboardGenerator(os.path.join(self.output_dir, "dashboards"))
        
        # Player database
        self.player_profiles = {}
        self._load_player_profiles()
        
        logger.info("ATLAS System initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            default_config = {
                "data_directory": "data",
                "model_directory": "models",
                "output_directory": "output",
                "sensor_config": {
                    "sampling_rate": 100,
                    "buffer_size": 1000
                },
                "processing_config": {
                    "filter_cutoff": 20,
                    "filter_order": 4
                }
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _load_player_profiles(self) -> None:
        """Load player profiles from storage."""
        profile_dir = os.path.join(self.data_dir, "profiles")
        
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
            return
        
        for profile_file in os.listdir(profile_dir):
            if profile_file.endswith(".json") and profile_file.startswith("profile_"):
                try:
                    profile_path = os.path.join(profile_dir, profile_file)
                    player_id = profile_file.replace("profile_", "").replace(".json", "")
                    
                    # Look for corresponding model file
                    model_filename = f"{player_id}.json"
                    model_path = os.path.join(profile_dir, model_filename)
                    
                    if os.path.exists(model_path):
                        profile = PlayerPerformanceProfile.load_profile(profile_path, model_path)
                    else:
                        profile = PlayerPerformanceProfile.load_profile(profile_path)
                    
                    self.player_profiles[player_id] = profile
                    logger.info(f"Loaded player profile for {player_id}")
                except Exception as e:
                    logger.error(f"Error loading profile {profile_file}: {e}")
    
    def setup_data_collection(self, session_name: str, session_type: str = "practice") -> DataCollectionSession:
        """
        Set up a data collection session.
        
        Args:
            session_name: Name for the session
            session_type: Type of session (practice or game)
            
        Returns:
            Configured DataCollectionSession
        """
        # Create session
        session = DataCollectionSession(session_name, session_type)
        
        # Set metadata (this would come from external sources in production)
        session.set_metadata({
            "field_conditions": "turf",
            "weather": "sunny",
            "temperature": 75,
            "humidity": 40,
            "notes": f"Data collection for {session_name}"
        })
        
        logger.info(f"Created data collection session: {session.session_id}")
        return session
    
    def add_player_to_session(self, 
                            session: DataCollectionSession, 
                            player_id: str, 
                            position: str) -> PlayerSensorSystem:
        """
        Add a player to a data collection session.
        
        Args:
            session: The data collection session
            player_id: ID of the player
            position: Position of the player
            
        Returns:
            Configured PlayerSensorSystem
        """
        # Create player sensor system
        player_system = PlayerSensorSystem(player_id, position)
        
        # Add sensors based on position
        if position.lower() == "quarterback":
            # QBs need full body tracking with emphasis on throwing arm
            helmet_imu = IMUSensor(f"H_{player_id}", 
                                   sampling_rate=100, 
                                   location="helmet")
            torso_imu = IMUSensor(f"T_{player_id}", 
                                 sampling_rate=100, 
                                 location="upper_back")
            r_arm_imu = IMUSensor(f"RA_{player_id}", 
                                 sampling_rate=200,  # Higher rate for throwing analysis
                                 location="right_upper_arm")
            r_forearm_imu = IMUSensor(f"RF_{player_id}", 
                                     sampling_rate=200, 
                                     location="right_forearm")
            
            player_system.add_sensor(helmet_imu)
            player_system.add_sensor(torso_imu)
            player_system.add_sensor(r_arm_imu)
            player_system.add_sensor(r_forearm_imu)
            
        elif "lineman" in position.lower():
            # Linemen need upper body and lower body tracking
            helmet_imu = IMUSensor(f"H_{player_id}", 
                                   sampling_rate=100, 
                                   location="helmet")
            torso_imu = IMUSensor(f"T_{player_id}", 
                                 sampling_rate=100, 
                                 location="upper_back")
            pelvis_imu = IMUSensor(f"P_{player_id}", 
                                  sampling_rate=100, 
                                  location="pelvis")
            
            player_system.add_sensor(helmet_imu)
            player_system.add_sensor(torso_imu)
            player_system.add_sensor(pelvis_imu)
            
        else:
            # Default sensor setup for other positions
            helmet_imu = IMUSensor(f"H_{player_id}", 
                                   sampling_rate=100, 
                                   location="helmet")
            torso_imu = IMUSensor(f"T_{player_id}", 
                                 sampling_rate=100, 
                                 location="upper_back")
            
            player_system.add_sensor(helmet_imu)
            player_system.add_sensor(torso_imu)
        
        # Add to session
        session.add_player_system(player_system)
        
        logger.info(f"Added player {player_id} ({position}) to session")
        return player_system
    
    def process_session_data(self, session_id: str) -> str:
        """
        Process data from a collection session.
        
        Args:
            session_id: ID of the session to process
            
        Returns:
            Path to processed data directory
        """
        session_data_path = os.path.join(self.data_dir, session_id)
        
        if not os.path.exists(session_data_path):
            raise ValueError(f"Session data not found: {session_id}")
        
        # Process the data
        processed_dir = self.data_pipeline.process_session_data(session_data_path)
        
        logger.info(f"Processed data for session {session_id}")
        return processed_dir
    
    def analyze_player_technique(self, 
                               player_id: str, 
                               technique_type: str, 
                               data_sequence: List[Dict]) -> Dict:
        """
        Analyze a player's technique from sensor data.
        
        Args:
            player_id: ID of the player
            technique_type: Type of technique to analyze
            data_sequence: Sensor data sequence covering the technique
            
        Returns:
            Analysis results
        """
        # Check if player profile exists, create if not
        if player_id not in self.player_profiles:
            # Need to determine position
            position = self._get_player_position(player_id, data_sequence)
            self.player_profiles[player_id] = PlayerPerformanceProfile(player_id, position)
        
        profile = self.player_profiles[player_id]
        
        # Create technique analyzer
        technique_analyzer = TechniqueAnalyzer()
        technique_analyzer.add_player_model(profile.biomechanical_model)
        
        # Analyze technique
        analysis_results = technique_analyzer.analyze_technique(technique_type, data_sequence)
        
        # Update player profile with results
        if "metrics" in analysis_results:
            # Average the metrics to get an overall score
            score_values = list(analysis_results["metrics"].values())
            if score_values:
                overall_score = sum(score_values) / len(score_values)
                timestamp = data_sequence[0].get("timestamp", time.time())
                profile.update_technique_score(technique_type, overall_score, timestamp)
        
        # Save updated profile
        profile_dir = os.path.join(self.data_dir, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile.save_profile(profile_dir)
        
        logger.info(f"Analyzed {technique_type} technique for player {player_id}")
        return analysis_results
    
    def assess_injury_risk(self, player_id: str) -> Dict:
        """
        Assess injury risk for a player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Risk assessment results
        """
        if player_id not in self.player_profiles:
            raise ValueError(f"Player profile not found: {player_id}")
        
        profile = self.player_profiles[player_id]
        
        # Calculate risk factors
        risk_assessment = profile.biomechanical_model.calculate_risk_factors()
        
        # Update profile
        profile.update_injury_risk(risk_assessment)
        
        # Save updated profile
        profile_dir = os.path.join(self.data_dir, "profiles")
        profile.save_profile(profile_dir)
        
        logger.info(f"Assessed injury risk for player {player_id}")
        return risk_assessment
    
    def generate_player_report(self, player_id: str, output_format: str = "dashboard") -> str:
        """
        Generate a comprehensive report for a player.
        
        Args:
            player_id: ID of the player
            output_format: Format for the report (dashboard or pdf)
            
        Returns:
            Path to generated report
        """
        if player_id not in self.player_profiles:
            raise ValueError(f"Player profile not found: {player_id}")
        
        profile = self.player_profiles[player_id]
        
        # Generate performance report from profile
        report_data = profile.generate_performance_report()
        
        # Create player data dictionary with all relevant info
        player_data = {
            "player_id": player_id,
            "position": profile.position,
            "performance_profile": {
                "technique_scores": profile.technique_scores,
                "injury_risk": profile.injury_risk,
                "anthropometrics": profile.biomechanical_model.anthropometrics
            },
            "recommendations": report_data.get("recommendations", []),
            "report_date": report_data.get("report_date", datetime.now().isoformat())
        }
        
        if output_format == "dashboard":
            # Generate interactive dashboard
            output_path = self.dashboard_generator.generate_player_dashboard(
                player_id, 
                player_data, 
                f"player_dashboard_{player_id}.html"
            )
        else:
            # TODO: Implement PDF report generation
            output_path = "PDF reports not implemented yet"
        
        logger.info(f"Generated report for player {player_id} in {output_format} format")
        return output_path
    
    def generate_team_report(self, team_name: str, player_ids: List[str]) -> str:
        """
        Generate a team report.
        
        Args:
            team_name: Name of the team
            player_ids: List of player IDs to include
            
        Returns:
            Path to generated report
        """
        # Collect data for all players
        team_data = {}
        
        for player_id in player_ids:
            if player_id in self.player_profiles:
                profile = self.player_profiles[player_id]
                
                # Create player data dictionary
                player_data = {
                    "player_id": player_id,
                    "position": profile.position,
                    "performance_profile": {
                        "technique_scores": profile.technique_scores,
                        "injury_risk": profile.injury_risk,
                        "anthropometrics": profile.biomechanical_model.anthropometrics
                    }
                }
                
                team_data[player_id] = player_data
        
        if not team_data:
            raise ValueError("No valid player profiles found")
        
        # Generate team dashboard
        output_path = self.dashboard_generator.generate_team_dashboard(
            team_data,
            f"team_dashboard_{team_name}.html"
        )
        
        logger.info(f"Generated team report for {team_name} with {len(team_data)} players")
        return output_path
    
    def train_models(self, training_data_path: str) -> Dict:
        """
        Train machine learning models with collected data.
        
        Args:
            training_data_path: Path to training data directory
            
        Returns:
            Training results
        """
        # This is a placeholder for actual training implementation
        # In a real system, would load and prepare training data
        
        logger.info(f"Model training initiated with data from {training_data_path}")
        return {"status": "Model training would be implemented here"}
    
    def _get_player_position(self, player_id: str, data_sequence: List[Dict]) -> str:
        """
        Determine player position from data or external source.
        This is a placeholder - in a real system would use a lookup or ML.
        """
        # In a real implementation, would:
        # 1. Look up in a player database
        # 2. Or use ML to classify position from movement patterns
        
        # For now, extract from data if available
        for frame in data_sequence:
            if "player_id" in frame and frame["player_id"] == player_id:
