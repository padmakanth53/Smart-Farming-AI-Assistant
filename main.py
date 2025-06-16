#!/usr/bin/env python3
"""
Smart Farming AI Assistant - Main Entry Point
Author: Smart Farm AI Team
Python Version: 3.10+
Compatible with PyCharm 2023
"""

import os
import sys
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
import json
from PIL import Image
import warnings
import base64
from io import BytesIO
import random  # Required for random simulations in SoilAnalysisAI
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Required for SoilAnalysisAI

warnings.filterwarnings('ignore')

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('farming_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Define model paths (assuming 'models' folder is in the project root)
SOIL_TYPE_MODEL_PATH = project_root / 'models' / 'soil_type_model.h5'
SOIL_TYPE_LABELS_PATH = project_root / 'models' / 'soil_type_labels.json'

# main.py - Replace your existing SoilAnalysisAI class with this
class SoilAnalysisAI:
    def __init__(self):
        logger.info("Initializing SoilAnalysisAI...")
        self.soil_type_model = None
        self.soil_type_labels = None
        self.use_demo_mode = False # Assume real mode by default, set to True if model loading fails

        try:
            # Load soil type model
            self.soil_type_model = load_model(SOIL_TYPE_MODEL_PATH)
            logger.info(f"Loaded Soil Type Model from {SOIL_TYPE_MODEL_PATH}")

            # Load soil type labels (mapping index to class name)
            with open(SOIL_TYPE_LABELS_PATH, 'r') as f:
                class_indices = json.load(f)
                # Invert the dictionary to map index (e.g., 0) to label (e.g., 'Clay')
                self.soil_type_labels = {int(v): k for k, v in class_indices.items()}
            logger.info(f"Loaded Soil Type Labels from {SOIL_TYPE_LABELS_PATH}")

        except Exception as e:
            logger.error(f"Error loading soil models: {e}. Soil analysis will use random simulation as fallback.", exc_info=True)
            self.use_demo_mode = True # Fallback to demo mode if models fail to load

        # Define NPK ranges based on soil type (scientifically typical averages or midpoints)
        # These are NOT random; they are fixed values representative of each soil type.
        self.npk_values_by_soil_type = {
            'Alluvial soil': {'N': 35.0, 'P': 25.0, 'K': 65.0}, # Example averages
            'Clayey soils': {'N': 40.0, 'P': 20.0, 'K': 75.0},
            'Laterite soil': {'N': 25.0, 'P': 15.0, 'K': 35.0},
            'Loamy soil': {'N': 30.0, 'P': 28.0, 'K': 55.0},
            'Sandy loam': {'N': 20.0, 'P': 12.0, 'K': 30.0},
            'Sandy soil': {'N': 15.0, 'P': 8.0, 'K': 25.0},
            # Add more soil types as per your dataset's labels, ensure they match!
            # For unknown/default:
            'Unknown': {'N': 30.0, 'P': 20.0, 'K': 40.0}
        }


    def _preprocess_image(self, image_data):
        """Converts base64 image data to a numpy array suitable for model input."""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            # Resize to model's expected input size and normalize
            image = image.resize((224, 224)) # Match IMG_HEIGHT, IMG_WIDTH from training
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
            img_array = img_array / 255.0 # Normalize pixel values to [0, 1]
            return img_array
        except Exception as e:
            logger.error(f"Error pre-processing image for soil analysis: {e}")
            return None

    def _decode_soil_type(self, prediction_array):
        """Decodes model's softmax output (probabilities) to soil type label."""
        if self.soil_type_labels is None:
            logger.warning("Soil type labels not loaded. Returning 'Unknown'.")
            return "Unknown" # Should not happen if model loaded correctly

        predicted_class_idx = np.argmax(prediction_array)
        return self.soil_type_labels.get(predicted_class_idx, "Unknown")

    def _get_npk_from_soil_type(self, soil_type):
        """
        Retrieves fixed NPK values based on the predicted soil type.
        This removes randomness and gives consistent, typical NPK values for each soil type.
        """
        npk = self.npk_values_by_soil_type.get(soil_type, self.npk_values_by_soil_type['Unknown'])
        return npk['N'], npk['P'], npk['K']


    def _get_fertilizer_recommendation(self):
        """
        Provides fertilizer recommendations based on predicted NPK values and soil type.
        This part uses the deterministic NPK values.
        """
        recommendations = []
        # NPK ranges (adjust these thresholds based on typical values for your crops/region)
        if self.N < 20: # Example threshold
            recommendations.append("Nitrogen is low. Apply nitrogen-rich fertilizers (e.g., Urea, Ammonium Sulfate) or organic compost like poultry manure.")
        elif self.N > 40:
            recommendations.append("Nitrogen is high. Avoid excess nitrogen; it can lead to leafy growth at the expense of fruit/flower development.")

        if self.P < 15: # Example threshold
            recommendations.append("Phosphorus is low. Consider phosphorus supplements (e.g., DAP, Rock Phosphate) for root development.")
        elif self.P > 30:
            recommendations.append("Phosphorus is high. While less common, excess P can tie up other nutrients like Zinc and Iron.")

        if self.K < 30: # Example threshold
            recommendations.append("Potassium is low. Add potassium-rich fertilizers (e.g., Muriate of Potash, Sulfate of Potash) or wood ash for fruit/flower quality.")
        elif self.K > 60:
            recommendations.append("Potassium is high. Usually not harmful but can affect the uptake of Calcium and Magnesium.")

        if not recommendations:
            recommendations.append("Your soil nutrients are well-balanced for general crop health.")

        # Add soil type specific recommendations
        if self.soil_type == 'Clayey soils':
            recommendations.append("Clayey soils have good nutrient retention but can be heavy. Incorporate organic matter to improve drainage and aeration.")
        elif self.soil_type == 'Sandy soil':
            recommendations.append("Sandy soils drain quickly, leading to nutrient leaching. Apply fertilizers in smaller, more frequent doses and incorporate organic matter to improve water retention.")
        elif self.soil_type == 'Loamy soil':
            recommendations.append("Loamy soil is ideal for most crops due to balanced drainage and nutrient retention. Maintain organic matter levels.")
        elif self.soil_type == 'Laterite soil':
            recommendations.append("Laterite soils are often acidic and infertile. Amend with lime to adjust pH and add organic matter regularly.")
        # ... add more soil type specific advice based on your dataset's types

        return recommendations

    def analyze_image(self, image_data):
        """Analyzes soil image to predict type and infer NPK values."""
        if self.use_demo_mode:
            logger.info("Soil Analysis AI running in demo mode (models not loaded).")
            # Fallback to general averages if models failed to load
            self.N = self.npk_values_by_soil_type['Unknown']['N']
            self.P = self.npk_values_by_soil_type['Unknown']['P']
            self.K = self.npk_values_by_soil_type['Unknown']['K']
            self.soil_type = "Unknown (Demo Mode)"
        else:
            processed_img = self._preprocess_image(image_data)
            if processed_img is None:
                # Return error if image processing failed
                return {"error": "Failed to process image for soil analysis. Invalid image data."}

            # Predict soil type
            soil_type_prediction = self.soil_type_model.predict(processed_img)
            self.soil_type = self._decode_soil_type(soil_type_prediction[0])
            logger.info(f"Predicted Soil Type: {self.soil_type}")

            # Infer NPK based on predicted soil type (deterministic values)
            self.N, self.P, self.K = self._get_npk_from_soil_type(self.soil_type)
            logger.info(f"Inferred NPK: N={self.N}, P={self.P}, K={self.K}")

        self.fertilizer_recommendations = self._get_fertilizer_recommendation()

        # Ensure the output structure matches the frontend's expectation
        return {
            "N": self.N,
            "P": self.P,
            "K": self.K,
            "soil_type": self.soil_type,
            "recommendations": self.fertilizer_recommendations
        }

# ... (Keep the rest of your main.py file as is, including the app initialization and run block)
# Ensure soil_analysis_ai instance is created after the class definition
soil_analysis_ai = SoilAnalysisAI() # This line should be present and remain unchanged

# You can update the print message in the main execution block for clarity:
# print("   🌱 Soil Analysis & Nutrient Prediction (AI-Powered)")
# Performance monitoring (currently unused in app flow, kept for future expansion)
class ModelPerformance:
    """Monitor and evaluate model performance"""

    def __init__(self):
        self.accuracy_threshold = 0.85
        self.confidence_threshold = 0.75

    def evaluate_prediction(self, prediction, confidence):
        """Evaluate individual prediction quality"""
        if confidence > self.confidence_threshold:
            return "High confidence"
        elif confidence > 0.5:
            return "Medium confidence"
        else:
            return "Low confidence - manual review recommended"

    def log_prediction(self, image_path, prediction, confidence):
        """Log predictions for performance analysis"""
        log_entry = {
            'timestamp': pd.Timestamp.now(),
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence
        }
        # In a real system, you would save this to a database or a persistent log file
        logger.info(f"Prediction logged: {log_entry}")


def initialize_app():
    """
    Initializes and returns the Flask application instance.
    This function is imported and called by main.py.
    """
    try:
        from app import app as flask_app_instance
        logger.info("Flask application instance initialized successfully.")
        return flask_app_instance
    except ImportError as e:
        logger.critical(f"Failed to import Flask app: {e}. Ensure 'app.py' exists and is correct.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error during Flask app initialization: {e}")
        sys.exit(1)


def check_dependencies():
    """Checks if all required Python packages are installed."""
    required_packages = [
        'flask', 'requests', 'Pillow', 'numpy', 'opencv-python',
        'tensorflow', 'googletrans', 'SpeechRecognition', 'pandas', 'sklearn'  # sklearn maps to scikit-learn
    ]
    # For pyaudio, it's often a system-level dependency or requires specific wheels.
    # It's less directly imported but critical for SpeechRecognition's audio capabilities.
    # We won't add it to this simple check but keep in mind for user troubleshooting.

    missing_packages = []
    for package in required_packages:
        try:
            # Special handling for scikit-learn
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing Python packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        sys.exit(1)
    else:
        logger.info("All required Python packages seem to be installed.")


if __name__ == '__main__':
    check_dependencies()
    logger.info("Loading Flask application...")
    # Import app here to avoid circular import if app.py also imports something from main.py at top level
    from app import app  # Now that check_dependencies is done, app should be safe to import

    if not app:  # This check is redundant if `from app import app` succeeds
        logger.critical("Failed to load Flask application instance. Exiting.")
        sys.exit(1)

    # Start the server
    try:
        logger.info("Starting Smart Farming AI Assistant Server...")
        print("\n🚀 Server Details:")
        print(f"   📡 Host: 0.0.0.0")
        print(f"   🔌 Port: 5000")
        print(f"   🌐 URL: http://localhost:5000")
        print(f"   📊 Debug Mode: ON")
        print("\n📋 Available Features:")
        print("   🌡️  Weather Monitoring & Alerts (with Geolocation)")
        print("   🌱 Soil Analysis & Nutrient Prediction (with CNN model simulation)")  # Updated
        print("   🐛 Pest Detection & Control (Simulated)")
        print("   💧 Water Level Monitoring (Simulated)")
        print("   🎤 Multi-language Voice Assistant")
        print("   🛒 Agri-Supplies Database")
        print("   📸 Mobile Camera & Drag-and-Drop Support")
        print("\n✅ System ready! Access the application at: http://localhost:5000")
        print("=" * 70)

        # Run the Flask application
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False  # Disable reloader to prevent duplicate startup issues
        )

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        print("\n🛑 Server shutting down gracefully...")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during server startup: {e}")
        sys.exit(1)

