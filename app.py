from flask import Flask, request, jsonify, render_template
import os
import base64
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
from datetime import datetime, timedelta
import requests
from googletrans import Translator
import speech_recognition as sr
from weather import WeatherService # Change made here: WeatherAPI changed to WeatherService
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize services
weather_api = WeatherService() # Change made here: WeatherAPI() changed to WeatherService()
translator = Translator()

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add this route to your app.py file  <--- ADD IT HERE
@app.route('/')
def index():
    return render_template('index.html')


# Demo data for fertilizers with real shop availability
FERTILIZER_DATABASE = {
    "ORGANIC": [
        {"name": "Cow Dung Manure", "price": "₹25/kg", "shop": "Kisan Depot", "phone": "+91-9876543210"},
        {"name": "Vermicompost", "price": "₹35/kg", "shop": "Agri Store", "phone": "+91-9876543211"},
        {"name": "Neem Cake", "price": "₹40/kg", "shop": "Green Farm Supplies", "phone": "+91-9876543212"},
        {"name": "Bone Meal", "price": "₹45/kg", "shop": "Organic Outlet", "phone": "+91-9876543213"}
    ],
    "CHEMICAL": [
        {"name": "Urea (46% N)", "price": "₹6/kg", "shop": "Farmers Choice", "phone": "+91-9876543214"},
        {"name": "DAP (18-46-0)", "price": "₹30/kg", "shop": "Agri Mart", "phone": "+91-9876543215"},
        {"name": "MOP (60% K2O)", "price": "₹20/kg", "shop": "Kisan Seva", "phone": "+91-9876543216"},
        {"name": "NPK (10-26-26)", "price": "₹28/kg", "shop": "Farm Solutions", "phone": "+91-9876543217"}
    ],
    "PESTICIDES": [
        {"name": "Neem Oil", "price": "₹180/liter", "shop": "Bio Pesticides Co", "phone": "+91-9876543218"},
        {"name": "Imidacloprid 17.8%", "price": "₹320/liter", "shop": "Crop Protection Ltd", "phone": "+91-9876543219"},
        {"name": "Chlorpyrifos 20%", "price": "₹280/liter", "shop": "Pest Control Hub", "phone": "+91-9876543220"}
    ]
}

# Crop-specific recommendations
CROP_RECOMMENDATIONS = {
    "rice": {"N": 120, "P": 60, "K": 40, "optimal_ph": 6.5},
    "wheat": {"N": 120, "P": 60, "K": 40, "optimal_ph": 7.0},
    "maize": {"N": 150, "P": 75, "K": 50, "optimal_ph": 6.8},
    "tomato": {"N": 100, "P": 50, "K": 80, "optimal_ph": 6.5},
    "cotton": {"N": 160, "P": 80, "K": 80, "optimal_ph": 7.5}
}
# app.py - After translator = Translator()
# Example: FERTILIZER_DATABASE (keep this as you had it)
# FERTILIZER_DATABASE = { ... }

# Initialize Pest/Disease Model (Global variables)
pest_disease_model = None
pest_disease_labels = None
try:
    # Assuming models folder is in project root, relative to app.py
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    PEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'pest_disease_model.h5')
    PEST_LABELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'pest_disease_labels.json')

    pest_disease_model = load_model(PEST_MODEL_PATH)
    with open(PEST_LABELS_PATH, 'r') as f:
        class_indices = json.load(f)
        # Invert the dictionary to map index (e.g., 0) to label (e.g., 'Aphids')
        pest_disease_labels = {int(v): k for k, v in class_indices.items()}

    logger.info(f"Loaded Pest/Disease Detection Model from {PEST_MODEL_PATH}")
    logger.info(f"Pest/Disease Labels: {pest_disease_labels}")
except Exception as e:
    logger.error(f"Error loading pest/disease detection model: {e}. Pest detection will be simulated as fallback.", exc_info=True)
    pest_disease_model = None # Set to None to trigger simulation logic

@app.route('/api/weather', methods=['GET'])
def get_weather():
    try:
        location = request.args.get('location', 'Vellore,Tamil Nadu')
        weather_data = weather_api.get_current_weather(location)

        # Check for weather alerts
        alerts = check_weather_alerts(weather_data)

        return jsonify({
            'success': True,
            'data': weather_data,
            'alerts': alerts
        })
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/soil-analysis', methods=['POST'])
def analyze_soil():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        file = request.files['image']
        crop_type = request.form.get('crop_type', 'general')

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Process image
        image = Image.open(file.stream)
        analysis_result = analyze_soil_image(image, crop_type)

        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'fertilizer_recommendations': get_fertilizer_recommendations(analysis_result, crop_type)
        })

    except Exception as e:
        logger.error(f"Soil analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# app.py - Replace your existing /api/pest-detection route with this
@app.route('/api/pest-detection', methods=['POST'])
def pest_detection():
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    # Extract base64 image data
    # Assuming image data comes as "data:image/jpeg;base64,..."
    image_data = request.json['image'].split(',')[1]

    pest_name = "Detection Unavailable"
    control_measures = "Please check the server logs for model loading errors. Ensure models are trained and saved correctly."

    if pest_disease_model: # Check if the model was successfully loaded
        try:
            # Pre-process image for model input
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((224, 224)) # Match model input size (IMG_HEIGHT, IMG_WIDTH from training)
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
            img_array = img_array / 255.0 # Normalize pixel values to [0, 1]

            # Perform prediction
            predictions = pest_disease_model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0]) # Get the index of the highest probability

            # Decode the predicted class index to the actual pest/disease name
            # Use .get() with a default in case an index isn't found (shouldn't happen with correct labels)
            pest_name = pest_disease_labels.get(predicted_class_idx, "Unknown Pest/Disease")
            logger.info(f"Detected Pest/Disease: {pest_name}")

            # Provide control measures based on the detected pest/disease
            # You MUST expand this dictionary/logic based on the actual classes in YOUR dataset
            # from the "Dataset for Crop Pest and Disease Detection"
            if "Healthy" in pest_name:
                control_measures = "Your crop appears healthy! Continue good farming practices."
            elif "Apple___Apple_scab" in pest_name: # Example from Plant Village-like datasets
                control_measures = "Detected Apple Scab. Remove infected leaves, prune for air circulation, apply fungicides (e.g., captan, myclobutanil)."
            elif "Tomato___Bacterial_spot" in pest_name:
                control_measures = "Detected Bacterial Spot on Tomato. Use disease-free seeds/transplants, copper-based sprays, and ensure good drainage."
            elif "Corn_(maize)___Common_rust" in pest_name:
                control_measures = "Detected Common Rust on Corn. Use resistant varieties, rotate crops, and consider fungicides in severe cases."
            elif "Potato___Late_blight" in pest_name:
                control_measures = "Detected Potato Late Blight. Remove infected plants, use certified seed, apply fungicides (e.g., chlorothalonil, mancozeb)."
            elif "Aphid" in pest_name: # If your dataset has specific pests
                control_measures = "Detected Aphids. Use insecticidal soap, neem oil, or introduce natural predators."
            # Add many more specific conditions based on your dataset's labels!
            else:
                control_measures = f"Detected {pest_name}. Consult local agricultural experts for specific control measures."

        except Exception as e:
            logger.error(f"Error during pest/disease detection inference: {e}", exc_info=True)
            pest_name = "Detection Error"
            control_measures = "Could not process image for pest/disease detection. Please try again or check server logs."
    else:
        logger.info("Pest/Disease detection running in simulation mode (model not loaded).")
        # Fallback to a generalized simulated response if model failed to load
        pest_name = "Detection Disabled (Model Not Loaded)"
        control_measures = "AI model for pest/disease detection is not loaded. Ensure training scripts were run successfully."


    return jsonify({
        "pest_name": pest_name,
        "control_measures": control_measures
        # Ensure the output keys match what your frontend expects: 'pest_name' and 'control_measures'
    })

# ... (Keep the rest of your app.py file as is)


@app.route('/api/voice-query', methods=['POST'])
def process_voice_query():
    try:
        data = request.json
        query = data.get('query', '')
        language = data.get('language', 'en')

        # Translate query to English if needed
        if language != 'en':
            query = translator.translate(query, src=language, dest='en').text

        # Process farming query
        response = process_farming_query(query)

        # Translate response back to requested language
        if language != 'en':
            response = translator.translate(response, src='en', dest=language).text

        return jsonify({
            'success': True,
            'response': response,
            'query': query
        })

    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/water-threshold', methods=['GET'])
def check_water_threshold():
    try:
        # Simulate water level monitoring
        import random
        water_level = random.randint(30, 95)

        status = "normal"
        message = "Water level is optimal"

        if water_level < 40:
            status = "critical"
            message = "Critical: Immediate irrigation required"
        elif water_level < 60:
            status = "warning"
            message = "Warning: Water level is low"

        return jsonify({
            'success': True,
            'water_level': water_level,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Water threshold error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


def analyze_soil_image(image, crop_type):
    """Analyze soil image and return nutrient analysis"""
    # Convert PIL image to numpy array for analysis
    img_array = np.array(image)

    # Simulate soil analysis based on image characteristics
    # In real implementation, use ML models for actual analysis
    avg_color = np.mean(img_array, axis=(0, 1))

    # Simulate analysis based on color characteristics
    soil_data = {
        'ph_level': round(6.0 + (avg_color[0] / 255) * 2, 1),
        'nitrogen': round(20 + (avg_color[1] / 255) * 60, 1),
        'phosphorus': round(15 + (avg_color[2] / 255) * 45, 1),
        'potassium': round(100 + (sum(avg_color) / 765) * 150, 1),
        'organic_matter': round(2.0 + (np.std(img_array) / 255) * 4, 1),
        'moisture': round(30 + (avg_color[2] / 255) * 40, 1),
        'soil_type': classify_soil_type(avg_color)
    }

    return soil_data


def classify_soil_type(avg_color):
    """Classify soil type based on color analysis"""
    r, g, b = avg_color

    if r > 150 and g > 100 and b < 100:
        return "Clay"
    elif r > 180 and g > 150 and b > 120:
        return "Sandy"
    elif r > 120 and g > 80 and b < 80:
        return "Loamy"
    else:
        return "Silt"


def detect_pest_in_image(image):
    """Detect pest in image and return pest information"""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Simulate pest detection based on image characteristics
    # In real implementation, use trained ML models

    pest_types = [
        {
            'name': 'Aphids',
            'scientific_name': 'Myzus persicae',
            'severity': 'Moderate',
            'confidence': 0.85,
            'description': 'Small green insects that feed on plant sap'
        },
        {
            'name': 'Whitefly',
            'scientific_name': 'Bemisia tabaci',
            'severity': 'High',
            'confidence': 0.78,
            'description': 'Small white flying insects that damage leaves'
        },
        {
            'name': 'Spider Mites',
            'scientific_name': 'Tetranychus urticae',
            'severity': 'Moderate',
            'confidence': 0.72,
            'description': 'Tiny mites that cause yellowing of leaves'
        }
    ]

    # Simulate selection based on image characteristics
    selected_pest = pest_types[hash(str(img_array.mean())) % len(pest_types)]

    return selected_pest


def get_pest_treatment(pest_info):
    """Get treatment recommendations for detected pest"""
    treatments = {
        'Aphids': [
            {'type': 'Organic', 'name': 'Neem Oil', 'dosage': '2-3ml/liter', 'frequency': 'Weekly'},
            {'type': 'Chemical', 'name': 'Imidacloprid', 'dosage': '0.3ml/liter', 'frequency': 'Bi-weekly'}
        ],
        'Whitefly': [
            {'type': 'Organic', 'name': 'Sticky Traps', 'dosage': '1 trap/10 plants', 'frequency': 'Replace weekly'},
            {'type': 'Chemical', 'name': 'Acetamiprid', 'dosage': '0.2g/liter', 'frequency': 'Bi-weekly'}
        ],
        'Spider Mites': [
            {'type': 'Organic', 'name': 'Predatory Mites', 'dosage': '1000/hectare', 'frequency': 'Once'},
            {'type': 'Chemical', 'name': 'Abamectin', 'dosage': '1ml/liter', 'frequency': 'Weekly'}
        ]
    }

    return treatments.get(pest_info['name'], [])


def get_fertilizer_recommendations(soil_data, crop_type):
    """Get fertilizer recommendations based on soil analysis and crop type"""
    recommendations = []

    # Get crop-specific requirements
    crop_req = CROP_RECOMMENDATIONS.get(crop_type, CROP_RECOMMENDATIONS['rice'])

    # Calculate deficiencies
    n_deficit = max(0, crop_req['N'] - soil_data['nitrogen'])
    p_deficit = max(0, crop_req['P'] - soil_data['phosphorus'])
    k_deficit = max(0, crop_req['K'] - soil_data['potassium'])

    # Recommend fertilizers based on deficiencies
    if n_deficit > 20:
        recommendations.extend([f for f in FERTILIZER_DATABASE['CHEMICAL'] if 'Urea' in f['name']])

    if p_deficit > 15:
        recommendations.extend([f for f in FERTILIZER_DATABASE['CHEMICAL'] if 'DAP' in f['name']])

    if k_deficit > 10:
        recommendations.extend([f for f in FERTILIZER_DATABASE['CHEMICAL'] if 'MOP' in f['name']])

    # Always recommend organic options
    recommendations.extend(FERTILIZER_DATABASE['ORGANIC'][:2])

    return recommendations


def process_farming_query(query):
    """Process farming-related queries and return relevant responses"""
    query_lower = query.lower()

    responses = {
        'water': "For optimal irrigation: Water early morning (6-8 AM) to minimize evaporation. Check soil moisture 2-3 inches deep. Drip irrigation saves 30-50% water compared to flood irrigation.",

        'fertilizer': "Use balanced NPK fertilizers based on soil test. Organic options: Compost (5-10 tons/hectare), Vermicompost (2-3 tons/hectare). Chemical: Urea for nitrogen, DAP for phosphorus, MOP for potassium.",

        'pest': "Integrated Pest Management (IPM) approach: Regular monitoring, biological control with beneficial insects, neem oil spray (2ml/liter), sticky traps for flying pests. Chemical pesticides only when necessary.",

        'disease': "Prevent plant diseases: Ensure proper spacing, good drainage, crop rotation. Remove infected plant parts immediately. Copper-based fungicides for bacterial diseases, systemic fungicides for fungal infections.",

        'weather': "Monitor weather forecasts daily. Protect crops from extreme weather: Use mulching in hot weather, drainage during heavy rains, wind barriers during storms. Plan operations according to weather windows.",

        'soil': "Maintain soil health: Regular addition of organic matter, avoid over-tillage, practice crop rotation, maintain proper pH (6.0-7.5 for most crops). Soil testing every 2-3 years is recommended.",

        'harvest': "Harvest at optimal maturity for best quality and yield. Early morning harvest provides better quality. Proper post-harvest handling reduces losses by 20-30%."
    }

    for keyword, response in responses.items():
        if keyword in query_lower:
            return response

    return "Thank you for your farming question! For specific advice, please provide details about your crop, location, and current growing conditions. Regular monitoring and proper planning are key to successful farming."


def check_weather_alerts(weather_data):
    """Check weather conditions and generate alerts"""
    alerts = []

    try:
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 50)
        wind_speed = weather_data.get('wind_speed', 10)
        rainfall = weather_data.get('rainfall', 0)

        if temp > 40:
            alerts.append({
                'type': 'warning',
                'message': 'High temperature alert! Ensure adequate irrigation and consider shade nets.',
                'icon': '🌡️'
            })

        if temp < 10:
            alerts.append({
                'type': 'warning',
                'message': 'Low temperature alert! Protect sensitive crops from frost damage.',
                'icon': '❄️'
            })

        if rainfall > 50:
            alerts.append({
                'type': 'warning',
                'message': 'Heavy rainfall expected! Ensure proper drainage to prevent waterlogging.',
                'icon': '🌧️'
            })

        if wind_speed > 30:
            alerts.append({
                'type': 'warning',
                'message': 'High wind alert! Secure tall crops and check for physical damage.',
                'icon': '💨'
            })

        if humidity > 85:
            alerts.append({
                'type': 'info',
                'message': 'High humidity may increase disease risk. Monitor crops closely.',
                'icon': '💧'
            })

        if not alerts:
            alerts.append({
                'type': 'success',
                'message': 'Weather conditions are favorable for farming activities.',
                'icon': '✅'
            })

    except Exception as e:
        logger.error(f"Weather alert error: {str(e)}")
        alerts.append({
            'type': 'info',
            'message': 'Weather monitoring active. Stay updated with local forecasts.',
            'icon': '🌤️'
        })

    return alerts


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)