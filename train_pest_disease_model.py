# train_pest_disease_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from pathlib import Path
import json
import os
import logging
import sys
from PIL import ImageFile, Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Pillow (to handle potentially truncated or large images) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200000000 # Set to 200 million pixels (200 MP)

# --- Paths Configuration ---
project_root = Path(__file__).parent

# IMPORTANT: Adjust 'PlantDoc' below if your dataset folder has a different name
# Assume PlantDoc dataset is structured like: your_project/data/PlantDoc/train/... and PlantDoc/test/...
DATASET_ROOT = project_root / 'data' / 'pest_disease_data'
TRAIN_DATA_DIR = DATASET_ROOT / 'train'
VAL_DATA_DIR = DATASET_ROOT / 'test' # Using 'test' directory for validation

MODEL_SAVE_PATH = project_root / 'models' / 'pest_disease_model.h5'
LABELS_SAVE_PATH = project_root / 'models' / 'pest_disease_labels.json'
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Standard input size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 5 # You might need more epochs for good performance

# Ensure 'models' directory exists
os.makedirs(project_root / 'models', exist_ok=True)

logger.info(f"--- Starting Pest/Disease Model Training ---")
logger.info(f"Loading training data from: {TRAIN_DATA_DIR}")
logger.info(f"Loading validation (test) data from: {VAL_DATA_DIR}")

# --- Data Loading and Augmentation ---
# For training data: apply augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation (test) data: only rescaling (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Use flow_from_directory to load images from the respective folders
# Ensure your dataset's 'train' and 'test' folders contain subfolders for each class.
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical' # Use 'categorical' for one-hot encoded labels
    )

    # Get class indices from training generator to ensure consistency
    # This is crucial for matching the output layer of the model
    # to the labels generated for both train and validation sets.
    classes_in_order = sorted(train_generator.class_indices.keys())

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=classes_in_order # <--- IMPORTANT CHANGE: Ensure validation uses same class order
    )
except Exception as e:
    logger.error(f"Error loading data with ImageDataGenerator: {e}")
    logger.error(f"Please ensure your dataset is correctly located at {DATASET_ROOT} and has 'train' and 'test' subfolders, each containing class subfolders, and that they contain images.")
    sys.exit(1)


num_classes = len(train_generator.class_indices) # This will be 28 based on your error
logger.info(f"Found {train_generator.num_classes} training images belonging to {num_classes} pest/disease types.")
logger.info(f"Found {validation_generator.num_classes} validation (test) images.")
logger.info(f"Detected {num_classes} pest/disease types: {train_generator.class_indices}")
logger.info(f"Validation generator class indices: {validation_generator.class_indices}") # Log validation indices for comparison

# Verify that the number of classes matches between train and validation generators
# This check will now likely pass if classes=... was the issue.
if len(train_generator.class_indices) != len(validation_generator.class_indices):
    logger.error("Mismatch in number of classes between training and validation sets AFTER forcing consistency!")
    logger.error(f"Training classes: {train_generator.class_indices}")
    logger.error(f"Validation classes: {validation_generator.class_indices}")
    logger.error("This indicates a deeper issue, possibly an empty class folder in the validation set which causes its count to drop despite forcing class order.")
    logger.error("Please ensure all class subfolders that exist in 'train' also exist in 'test' and contain at least one image.")
    sys.exit(1)


# Save the class_indices mapping (index to label)
try:
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    logger.info(f"Pest/Disease labels saved to {LABELS_SAVE_PATH}")
except Exception as e:
    logger.error(f"Error saving labels: {e}")

# --- Model Creation (Transfer Learning with MobileNetV2) ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # num_classes is derived from train_generator

model = Model(inputs=base_model.input, outputs=predictions)

# --- Model Compilation ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Model Training ---
logger.info("\nStarting Pest/Disease Model Training...")
try:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    logger.info("Pest/Disease Model Training Complete.")
except Exception as e:
    logger.error(f"Error during model training: {e}")
    logger.error("A common cause is corrupted images within the dataset. Try running check_images.py again.")
    sys.exit(1)

# --- Model Evaluation ---
logger.info("\nEvaluating Pest/Disease Detection Model...")
try:
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
except Exception as e:
    logger.error(f"Error during model evaluation: {e}")

# --- Model Saving ---
logger.info(f"\nSaving Pest/Disease Detection Model to {MODEL_SAVE_PATH}...")
try:
    model.save(MODEL_SAVE_PATH)
    logger.info("Model saved successfully.")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    logger.error("You might want to check disk space or permissions.")

print("\nScript finished.")