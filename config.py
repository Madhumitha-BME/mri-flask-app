# config.py

import os

# --- Data Paths ---
# If you have a full dataset with multiple patient folders:
# DATA_DIR = "path/to/your/brats_dataset_root"

# For your specific single patient folder:
# config.py

# --- Data Paths ---
# If you have a full dataset with multiple patient folders:
DATA_DIR = r"C:\Users\COLLEGE\mri_segmentation\data" # Points to the folder containing all BraTSXX_Training_YYY folders

# For your specific single patient folder (commented out for multi-patient training):
# SINGLE_PATIENT_DATA_DIR = r"C:\Users\COLLEGE\mri_segmentation\data\BraTS20_Training_032"
# Note the 'r' prefix for a raw string to handle backslashes correctly.
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)


# Modalities and mask suffix (adjust based on your BraTS file naming)
MODALITIES_TO_USE = ['t1']
MASK_FILE_SUFFIX = '_seg' # e.g., BraTS20_Training_032_seg.nii.gz

# --- Image Processing Parameters ---
IMG_DIM = (128, 128, 128) # Target dimensions for processed 3D volumes
NUM_CHANNELS = len(MODALITIES_TO_USE) # This will be 1 for the test

TARGET_VOXEL_SPACING = (1.0, 1.0, 1.0) # Target isotropic spacing (mm)

# --- Model Parameters ---
NUM_CHANNELS = len(MODALITIES_TO_USE) # Automatically set based on modalities used
NUM_CLASSES = 1 # Binary segmentation (e.g., metastasis vs. background)

# --- Training Parameters ---
BATCH_SIZE = 1 # For single patient demo, use 1. For full dataset, typically 2, 4, 8...
EPOCHS = 20 # Number of training epochs (increase significantly for real training)
LEARNING_RATE = 0.0001 # Adam optimizer learning rate
VALIDATION_SPLIT = 0.2 # Percentage of data for validation (only applies to multi-patient dataset)

# --- Output Paths ---
OUTPUT_MODEL_DIR = "trained_models"
PREDICTED_MASKS_DIR = "predicted_masks"

# Create output directories if they don't exist
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTED_MASKS_DIR, exist_ok=True)