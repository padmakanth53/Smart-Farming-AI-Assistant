# train_soil_type_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

# --- Configuration ---
# !!! IMPORTANT: Adjust DATA_DIR if your dataset structure is different (e.g., has a 'train' folder inside)
# If your images are directly in soil_type_data/Clay, soil_type_data/Loam, use:
DATA_DIR = 'data/soil_type_data'
# If your images are in soil_type_data/train/Clay, soil_type_data/train/Loam etc. AND soil_type_data/val/Clay, soil_type_data/val/Loam:
# TRAIN_DATA_DIR = 'data/soil_type_data/train'
# VAL_DATA_DIR = 'data/soil_type_data/validation' # or 'data/soil_type_data/val' or 'data/soil_type_data/test'
# Then adjust flow_from_directory calls below. For simplicity, we'll assume DATA_DIR contains class folders directly.


MODEL_SAVE_PATH = 'models/soil_type_model.h5'
LABELS_SAVE_PATH = 'models/soil_type_labels.json'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 15 # You might need more epochs (e.g., 20-50) for better accuracy

# Ensure 'models' directory exists
os.makedirs('models', exist_ok=True)

print(f"--- Starting Soil Type Model Training ---")
print(f"Loading data from: {DATA_DIR}")

# --- Data Loading and Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20, # Data Augmentation to prevent overfitting
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # For multi-class classification
    subset='training',
    seed=42 # For reproducibility
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} soil types: {train_generator.class_indices}")

# Save the class_indices mapping (index to label)
# This is CRUCIAL for decoding predictions in app.py
with open(LABELS_SAVE_PATH, 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f"Soil type labels saved to {LABELS_SAVE_PATH}")


# --- Model Creation (Transfer Learning with MobileNetV2) ---
# Load MobileNetV2 pre-trained on ImageNet, without its classification head
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, # Do not include the ImageNet classification layer
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3) # Specify input image dimensions (height, width, color channels)
)

# Freeze the base model layers: This means their weights will not be updated during training.
# This speeds up training and allows the pre-trained features to be utilized.
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions to a single vector per feature map
x = Dense(128, activation='relu')(x) # A dense (fully connected) layer with ReLU activation
predictions = Dense(num_classes, activation='softmax')(x) # Output layer: one neuron per class, softmax for probabilities

model = Model(inputs=base_model.input, outputs=predictions)

# --- Model Compilation ---
# Optimizer: Adam is a popular choice for deep learning
# Loss: Categorical Crossentropy for multi-class classification with one-hot encoded labels
# Metrics: Accuracy to monitor performance during training
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Model Training ---
print("\nStarting Soil Type Model Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1 # Show training progress
)

# --- Model Evaluation ---
# Evaluate the model on the validation set to get final performance metrics
print("\nEvaluating Soil Type Model...")
loss, accuracy = model.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# --- Model Saving ---
# Save the entire model (architecture, weights, optimizer state) to a .h5 file
model.save(MODEL_SAVE_PATH)
print(f"Soil Type Model saved to {MODEL_SAVE_PATH}")
print("\n--- Soil Type Model Training Complete! ---")