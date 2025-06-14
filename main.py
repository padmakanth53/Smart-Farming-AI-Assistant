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
import cv2
from PIL import Image
import warnings
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

# Define model path (conceptual, as we can't write to disk directly in this env)
MODEL_SAVE_PATH = 'soil_analysis_model.h5'


class SoilAnalysisAI:
    """
    Handles soil image analysis, predicting nutrient levels and soil type,
    and providing fertilizer recommendations.
    Includes a conceptual CNN model for demonstration.
    """

    def __init__(self):
        self.soil_analysis_model = None  # This will hold the combined CNN model
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Soil types for classification output
        self.soil_types = {
            0: 'Clay', 1: 'Sandy', 2: 'Silty', 3: 'Peaty', 4: 'Chalky', 5: 'Loamy'
        }
        self.num_soil_types = len(self.soil_types)

        # Fertilizer recommendations database with shop and phone details
        self.fertilizer_db = {
            'Clay': {
                'base_fertilizer': {'name': 'NPK 10-10-10', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Compost, Manure', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'drainage_improver': {'name': 'Sand, Perlite', 'shop': 'FarmSupply', 'phone': '+91-9876543212'},
                'ph_adjuster': {'name': 'Lime (if acidic)', 'shop': 'KisanBazaar', 'phone': '+91-9876543213'}
            },
            'Sandy': {
                'base_fertilizer': {'name': 'NPK 15-5-15', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Compost, Peat Moss', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'water_retention_improver': {'name': 'Vermiculite, Clay', 'shop': 'HydroFarm',
                                             'phone': '+91-9876543214'},
                'ph_adjuster': {'name': 'Sulphur (if alkaline)', 'shop': 'KisanBazaar', 'phone': '+91-9876543213'}
            },
            'Silty': {
                'base_fertilizer': {'name': 'NPK 8-12-10', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Compost, Green Manure', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'drainage_improver': {'name': 'Sand', 'shop': 'FarmSupply', 'phone': '+91-9876543212'},
                'ph_adjuster': {'name': 'Dolomite (if acidic)', 'shop': 'KisanBazaar', 'phone': '+91-9876543213'}
            },
            'Peaty': {
                'base_fertilizer': {'name': 'NPK 5-10-15', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Compost, Aged Bark', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'drainage_improver': {'name': 'Sand, Gravel', 'shop': 'FarmSupply', 'phone': '+91-9876543212'},
                'ph_adjuster': {'name': 'Lime (if acidic)', 'shop': 'KisanBazaar', 'phone': '+91-9876543213'}
            },
            'Chalky': {
                'base_fertilizer': {'name': 'NPK 12-8-10', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Compost, Manure', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'iron_supplement': {'name': 'Iron Chelate', 'shop': 'FertilizerHub', 'phone': '+91-9876543215'},
                'ph_adjuster': {'name': 'Sulphur (if alkaline)', 'shop': 'KisanBazaar', 'phone': '+91-9876543213'}
            },
            'Loamy': {
                'base_fertilizer': {'name': 'NPK 10-10-10', 'shop': 'Kisan Depot', 'phone': '+91-9876543210'},
                'organic_matter': {'name': 'Balanced Compost', 'shop': 'Agri Store', 'phone': '+91-9876543211'},
                'general_purpose': {'name': 'Well-drained and fertile soil, maintain with balanced fertilizers',
                                    'shop': 'FarmSupply', 'phone': '+91-9876543212'}
            }
        }
        # Initialize the model (conceptual loading/creation)
        self._initialize_model()

    def _initialize_model(self):
        """
        Initializes or "loads" the CNN model. In a real app, this would load a pre-trained model.
        Here, we create a dummy model to allow the code to run.
        """
        try:
            logger.info("Simulating loading pre-trained soil analysis model...")
            # In a real scenario: self.soil_analysis_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            self.soil_analysis_model = self.create_cnn_model()  # Create the model structure
            logger.info("Dummy soil analysis model created (simulated load).")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Creating a new one.")
            self.soil_analysis_model = self.create_cnn_model()
            # If a model needs training, it would conceptually happen here:
            # self.train_model(self.soil_analysis_model)
            # self.soil_analysis_model.save(MODEL_SAVE_PATH) # Simulate saving

    def create_cnn_model(self, input_shape=(128, 128, 3)):
        """
        Defines a strong CNN model for soil analysis.
        This model will have two outputs: one for soil type classification (softmax)
        and one for nutrient level regression (linear).
        """
        inputs = keras.Input(shape=input_shape)

        # Feature extraction layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)  # Global Average Pooling as requested

        # Output branch for Soil Type Classification
        # Using a Dense layer with softmax for multi-class classification
        soil_type_output = layers.Dense(self.num_soil_types, activation='softmax', name='soil_type_output')(x)

        # Output branch for Nutrient Level Regression (N, P, K)
        # Using a Dense layer with linear activation for regression
        nutrient_output = layers.Dense(3, activation='linear', name='nutrient_output')(x)  # 3 outputs for N, P, K

        # Define the model with multiple inputs/outputs
        model = keras.Model(inputs=inputs, outputs=[soil_type_output, nutrient_output], name='Soil_Analysis_CNN')
        logger.info("CNN model architecture defined.")
        return model

    def train_model(self, model):
        """
        Simulates the training process for the soil analysis model.
        In a real application, you would load your dataset (e.g., from `_simulate_soil_dataset`),
        preprocess it, and run the actual training loop here.
        """
        logger.info("Simulating model training...")
        # Placeholder for actual training setup
        model.compile(
            optimizer='adam',
            loss={'soil_type_output': 'categorical_crossentropy', 'nutrient_output': 'mse'},
            metrics={'soil_type_output': 'accuracy', 'nutrient_output': 'mae'}
        )

        # To run actual training, you would need real data.
        # Example of how you would call fit with dummy data:
        # dummy_images, dummy_soil_labels, dummy_nutrient_labels = self._simulate_soil_dataset()
        # model.fit(dummy_images, {'soil_type_output': dummy_soil_labels, 'nutrient_output': dummy_nutrient_labels}, epochs=10)

        logger.info("Model training simulation completed.")

    def _simulate_soil_dataset(self, num_samples=100):
        """
        Simulates a small dataset for demonstration purposes.
        In a real scenario, this would load actual images and labels from your dataset folder.
        """
        logger.info(f"Simulating a dataset of {num_samples} samples.")
        # Create dummy image data (e.g., 128x128 RGB images)
        dummy_images = np.random.rand(num_samples, 128, 128, 3).astype(np.float32)

        # Create dummy soil type labels (one-hot encoded)
        dummy_soil_labels = keras.utils.to_categorical(
            np.random.randint(0, self.num_soil_types, num_samples),
            num_classes=self.num_soil_types
        )

        # Create dummy nutrient labels (N, P, K in ppm)
        # Values are made somewhat realistic for demonstration
        dummy_nutrient_labels = np.random.rand(num_samples, 3).astype(np.float32) * [150, 70, 100] + [20, 10, 30]

        return dummy_images, dummy_soil_labels, dummy_nutrient_labels

    def preprocess_image_for_prediction(self, image_data_bytesio):
        """
        Preprocesses a BytesIO image object for model prediction.
        Resizes to 128x128 and normalizes pixel values.
        """
        # Convert BytesIO to PIL Image, then to NumPy array
        img = Image.open(image_data_bytesio).resize((128, 128))
        img_array = np.array(img).astype(np.float32)

        # Ensure image has 3 channels (convert grayscale to RGB if needed)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Discard alpha channel

        # Normalize pixel values to [0, 1]
        if img_array.max() > 1.0:
            img_array /= 255.0

        # Add batch dimension (model expects a batch of images)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_nutrients_from_image(self, image_data_bytesio):
        """
        Simulates prediction of N, P, K nutrient levels and soil type from a soil image.
        This function will use heuristics based on image properties to simulate realistic
        nutrient and soil type predictions, and then generate specific fertilizer recommendations.
        """
        try:
            # Preprocess the input image
            processed_image = self.preprocess_image_for_prediction(image_data_bytesio)

            # In a real scenario, you would run:
            # soil_type_probs, nutrient_levels = self.soil_analysis_model.predict(processed_image)
            # predicted_soil_type_idx = np.argmax(soil_type_probs[0])
            # predicted_soil_type = self.soil_types[predicted_soil_type_idx]
            # predicted_n, predicted_p, predicted_k = nutrient_levels[0]

            # For demonstration, we simulate plausible predictions:
            # Simple heuristic: "Darker" soil images might have more organic matter/nitrogen
            avg_pixel_value = np.mean(processed_image)  # Average pixel value across the image

            # Simulate nutrient levels (ppm) based on simple image characteristics
            # These are designed to sometimes trigger low levels for recommendations
            nitrogen = round(random.uniform(20, 100) * (1.2 - avg_pixel_value),
                             2)  # Tends to be higher for darker soils
            phosphorus = round(random.uniform(10, 60) * (1.0 + avg_pixel_value), 2)  # Varies
            potassium = round(random.uniform(30, 120) * (0.8 + avg_pixel_value * 0.5), 2)  # Varies

            # Simulate soil type based on a rough heuristic or random choice if image isn't clear
            if avg_pixel_value < 0.35:  # Darker soils
                soil_type = random.choice(['Clay', 'Peaty', 'Silty'])
            elif avg_pixel_value > 0.65:  # Lighter soils
                soil_type = random.choice(['Sandy', 'Chalky'])
            else:  # Medium brightness
                soil_type = random.choice(['Loamy', 'Silty', 'Clay'])

            # Ensure some nutrients are low sometimes to trigger specific recommendations
            if random.random() < 0.3:  # 30% chance Nitrogen is low
                nitrogen = round(random.uniform(10, 40), 2)
            if random.random() < 0.3:  # 30% chance Phosphorus is low
                phosphorus = round(random.uniform(5, 25), 2)
            if random.random() < 0.3:  # 30% chance Potassium is low
                potassium = round(random.uniform(20, 60), 2)

            # Generate fertilizer recommendations based on these simulated nutrient levels and soil type
            fertilizer_recommendations = self._get_fertilizer_recommendations(
                soil_type, nitrogen, phosphorus, potassium
            )

            return {
                'nitrogen_level': f"{nitrogen:.2f} ppm",
                'phosphorus_level': f"{phosphorus:.2f} ppm",
                'potassium_level': f"{potassium:.2f} ppm",
                'soil_type': soil_type,
                'fertilizer_recommendations': fertilizer_recommendations
            }
        except Exception as e:
            logger.error(f"Error predicting nutrients from image: {e}")
            return {
                'nitrogen_level': "N/A",
                'phosphorus_level': "N/A",
                'potassium_level': "N/A",
                'soil_type': "Unknown",
                'fertilizer_recommendations': []
            }

    def _get_fertilizer_recommendations(self, soil_type, n_level, p_level, k_level):
        """
        Generates fertilizer recommendations based on soil type and specific nutrient levels.
        Prioritizes specific deficiencies and then adds general soil type recommendations.
        """
        recommendations = []

        # Add recommendations for specific nutrient deficiencies
        # Assuming optimal levels (these would be from crop-specific data in a real app)
        optimal_n = 80
        optimal_p = 40
        optimal_k = 100

        if n_level < (optimal_n * 0.5):  # Very low nitrogen
            rec = {'name': 'Urea (Nitrogen)', 'reason': 'Very low Nitrogen detected. Apply Urea for quick boost.',
                   'shop': 'Kisan Depot', 'phone': '+91-9876543210'}
            recommendations.append(rec)
        elif n_level < optimal_n:  # Moderately low nitrogen
            rec = {'name': 'Ammonium Sulphate', 'reason': 'Low Nitrogen detected. Apply Ammonium Sulphate.',
                   'shop': 'AgriMart', 'phone': '+91-9876543214'}
            recommendations.append(rec)

        if p_level < (optimal_p * 0.5):  # Very low phosphorus
            rec = {'name': 'DAP (Phosphorus)', 'reason': 'Very low Phosphorus detected. DAP is highly recommended.',
                   'shop': 'FarmSolutions', 'phone': '+91-9876543217'}
            recommendations.append(rec)
        elif p_level < optimal_p:  # Moderately low phosphorus
            rec = {'name': 'Single Super Phosphate (SSP)', 'reason': 'Low Phosphorus detected. SSP can help.',
                   'shop': 'AgriStore', 'phone': '+91-9876543211'}
            recommendations.append(rec)

        if k_level < (optimal_k * 0.5):  # Very low potassium
            rec = {'name': 'MOP (Potassium)', 'reason': 'Very low Potassium detected. MOP for fruit development.',
                   'shop': 'Kisan Seva', 'phone': '+91-9876543216'}
            recommendations.append(rec)
        elif k_level < optimal_k:  # Moderately low potassium
            rec = {'name': 'SOP (Sulphate of Potash)', 'reason': 'Low Potassium detected. SOP is a good option.',
                   'shop': 'Organic Outlet', 'phone': '+91-9876543213'}
            recommendations.append(rec)

        # Add general soil type recommendations if not already covered by specific nutrient needs
        base_recs = self.fertilizer_db.get(soil_type, {})
        for key, details in base_recs.items():
            # Check if a similar fertilizer is already recommended for specific deficiency
            already_recommended = any(details['name'] in rec['name'] for rec in recommendations)
            if not already_recommended:
                recommendations.append({
                    'name': details['name'],
                    'reason': f"General recommendation for {soil_type} soil ({key.replace('_', ' ')}).",
                    'shop': details.get('shop', 'Local Agri Store'),
                    'phone': details.get('phone', 'N/A')
                })

        # Ensure uniqueness based on fertilizer name
        unique_recommendations = []
        seen_names = set()
        for rec in recommendations:
            if rec['name'] not in seen_names:
                unique_recommendations.append(rec)
                seen_names.add(rec['name'])

        return unique_recommendations


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

