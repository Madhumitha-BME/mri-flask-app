# predict.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import functions/classes from your src modules
from src.data_processor import preprocess_patient_data, save_nifti
from src.metrics import dice_coefficient, combined_loss
# Assuming display_overlay is in src/utils.py, but we'll make a new one for differences
from src.utils import display_overlay 

# Import configuration parameters
from config import (
    IMG_DIM, MODALITIES_TO_USE, MASK_FILE_SUFFIX, PREDICTED_MASKS_DIR,
    OUTPUT_MODEL_DIR, NUM_CLASSES
)

def calculate_metrics(y_true, y_pred, y_pred_probs):
    """
    Calculates and prints performance metrics for a single prediction.
    y_true: ground truth binary mask
    y_pred: predicted binary mask
    y_pred_probs: predicted probability mask
    """
    dice_score = dice_coefficient(tf.constant(y_true, dtype=tf.float32), 
                                  tf.constant(y_pred_probs, dtype=tf.float32)).numpy()

    y_true_tensor = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_tensor = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    true_positives = tf.reduce_sum(y_true_tensor * y_pred_tensor)
    false_positives = tf.reduce_sum((1 - y_true_tensor) * y_pred_tensor)
    false_negatives = tf.reduce_sum(y_true_tensor * (1 - y_pred_tensor))

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    iou = true_positives / (true_positives + false_positives + false_negatives + tf.keras.backend.epsilon())

    print("\n--- Performance Metrics ---")
    print(f"Dice Coefficient: {dice_score:.4f}")
    print(f"Intersection over Union (IoU): {iou.numpy():.4f}")
    print(f"Precision: {precision.numpy():.4f}")
    print(f"Recall (Sensitivity): {recall.numpy():.4f}")
    print(f"F1-Score: {f1_score.numpy():.4f}")
    print("---------------------------")


def display_difference(mri_slice, gt_slice, pred_slice, title="Segmentation Difference"):
    """
    Displays the difference between the ground truth and predicted masks.
    - True Positives (correctly predicted): White
    - False Positives (predicted but not ground truth): Red
    - False Negatives (ground truth but not predicted): Green
    """
    # Remove channel dim for slices
    gt_slice = np.squeeze(gt_slice)
    pred_slice = np.squeeze(pred_slice)

    # Create a single image showing the differences
    difference_image = np.zeros((*mri_slice.shape, 3), dtype=np.float32) # RGB image

    # True Positives (overlap, correctly predicted) - White
    difference_image[gt_slice > 0.5] = [1, 1, 1]

    # False Negatives (missed by the model) - Green
    difference_image[(gt_slice > 0.5) & (pred_slice < 0.5)] = [0, 1, 0]

    # False Positives (over-segmented by the model) - Red
    difference_image[(gt_slice < 0.5) & (pred_slice > 0.5)] = [1, 0, 0]

    # Display the original MRI slice
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[0].set_title("Original MRI Slice")
    ax[0].axis('off')

    # Overlay the difference map with the original image
    ax[1].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[1].imshow(difference_image.T, alpha=0.5, origin='lower') # Adjust alpha as needed
    ax[1].set_title(title)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    print("--- Starting Prediction with Trained U-Net Model ---")
    model_path = os.path.join(OUTPUT_MODEL_DIR, "brain_mets_unet_multi_patient_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please check your training script and paths.")
        return

    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'combined_loss': combined_loss
    }
    
    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    NEW_PATIENT_DATA_DIR = r"C:\Users\COLLEGE\mri_segmentation\data\BraTS20_Training_033"
    
    if not os.path.exists(NEW_PATIENT_DATA_DIR):
        print(f"Error: New patient data directory not found at {NEW_PATIENT_DATA_DIR}.")
        return

    print(f"\nProcessing patient data from: {NEW_PATIENT_DATA_DIR}")

    new_mri_processed, new_mask_gt, new_affine, new_header = preprocess_patient_data(
        NEW_PATIENT_DATA_DIR, IMG_DIM, MODALITIES_TO_USE, MASK_FILE_SUFFIX
    )
    
    if new_mri_processed is None:
        print("Failed to preprocess patient data. Exiting.")
        return
        
    X_new = np.expand_dims(new_mri_processed, axis=0)
    print(f"Preprocessed input shape for prediction: {X_new.shape}")

    print("\nPerforming segmentation prediction...")
    predicted_mask_probs = model.predict(X_new)[0]
    predicted_mask = (predicted_mask_probs > 0.5).astype(np.uint8)

    calculate_metrics(np.squeeze(new_mask_gt), np.squeeze(predicted_mask), np.squeeze(predicted_mask_probs))
    
    predicted_mask_squeezed = np.squeeze(predicted_mask)
    patient_id_folder = os.path.basename(NEW_PATIENT_DATA_DIR)
    output_prediction_file = f"{patient_id_folder}_predicted_seg.nii"
    output_prediction_path = os.path.join(PREDICTED_MASKS_DIR, output_prediction_file)
    save_nifti(predicted_mask_squeezed, new_affine, new_header, output_prediction_path)
    
    print("Prediction and metric calculation complete.")
    print(f"The predicted segmentation mask is saved at: {output_prediction_path}")
    
    
    # --- Visualization Section ---
    slice_to_view = IMG_DIM[2] // 2
    modality_to_view = 1 # T1ce
    
    # 1. Visualize the Ground Truth
    display_overlay(new_mri_processed, new_mask_gt, slice_to_view, axis=2,
                    title=f"Ground Truth for {patient_id_folder}", modality_channel=modality_to_view)
    
    # 2. Visualize the Predicted Mask
    display_overlay(new_mri_processed, predicted_mask, slice_to_view, axis=2,
                    title=f"Prediction for {patient_id_folder}", modality_channel=modality_to_view)

    # 3. Visualize the Difference (Error) Mask
    new_mri_slice = new_mri_processed[:,:,slice_to_view, modality_to_view]
    gt_slice = new_mask_gt[:,:,slice_to_view]
    pred_slice = predicted_mask[:,:,slice_to_view]
    display_difference(new_mri_slice, gt_slice, pred_slice, 
                       title=f"Error Map for {patient_id_folder} (TP=White, FN=Green, FP=Red)")


if __name__ == "__main__":
    main()