# train.py (FOR MULTIPLE PATIENTS)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob # Import glob to find directories

# Import functions/classes from your src modules
from src.data_processor import preprocess_patient_data, data_generator, save_nifti
from src.models import unet_3d
from src.metrics import dice_coefficient, combined_loss
from src.utils import display_overlay

# Import configuration parameters
from config import (
    DATA_DIR, MODALITIES_TO_USE, MASK_FILE_SUFFIX,
    IMG_DIM, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    OUTPUT_MODEL_DIR, PREDICTED_MASKS_DIR
)

def main():
    print("--- Starting Brain Metastasis Segmentation Training (Multi-Patient) ---")

    # --- Data Collection and Splitting ---
    # Find all patient directories within the main data directory
    # Assumes each patient folder name starts with "BraTS" and contains NIfTI files
    all_patient_dirs = sorted(glob(os.path.join(DATA_DIR, "BraTS*_Training_*")))

    if not all_patient_dirs:
        print(f"Error: No patient directories found in {DATA_DIR}. Please check DATA_DIR and folder naming.")
        return

    np.random.shuffle(all_patient_dirs) # Shuffle patient order
    num_patients = len(all_patient_dirs)
    num_train = int(num_patients * (1 - VALIDATION_SPLIT))
    train_patient_dirs = all_patient_dirs[:num_train]
    val_patient_dirs = all_patient_dirs[num_train:]

    print(f"Total patients found: {num_patients}")
    print(f"Training patients: {len(train_patient_dirs)}")
    print(f"Validation patients: {len(val_patient_dirs)}")

    # --- Create Data Generators ---
    # These generators will load and preprocess data on-the-fly, patient by patient,
    # and yield batches for training.
    train_generator = data_generator(
        train_patient_dirs, IMG_DIM, BATCH_SIZE, MODALITIES_TO_USE, MASK_FILE_SUFFIX
    )
    val_generator = data_generator(
        val_patient_dirs, IMG_DIM, BATCH_SIZE, MODALITIES_TO_USE, MASK_FILE_SUFFIX
    )

    # --- Build and Compile Model ---
    print("Building 3D U-Net model...")
    # The first patient's data is loaded to infer NUM_CHANNELS for model input_shape
    # (assuming all patients have the same number of modalities/channels)
    
    # Temporarily load one patient to get actual number of channels
    temp_img, _, _, _ = preprocess_patient_data(
        train_patient_dirs[0], IMG_DIM, MODALITIES_TO_USE, MASK_FILE_SUFFIX
    )
    if temp_img is None:
        print("Failed to load initial patient for channel determination. Exiting.")
        return
    
    actual_num_channels = temp_img.shape[-1]
    
    model = unet_3d(input_shape=(IMG_DIM[0], IMG_DIM[1], IMG_DIM[2], actual_num_channels),
                    num_classes=NUM_CLASSES)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient, 'accuracy'])
    model.summary()

    # --- Train Model ---
    print(f"Training model for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_patient_dirs) // BATCH_SIZE, # Ensure steps per epoch aligns with batch size
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_patient_dirs) // BATCH_SIZE,
        verbose=1
    )

    # --- Save Trained Model ---
    model_save_path = os.path.join(OUTPUT_MODEL_DIR, "brain_mets_unet_multi_patient_model.h5")
    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}")

    # --- Plot Training History ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coefficient'], label='Train Dice Coeff')
    if 'val_dice_coefficient' in history.history:
        plt.plot(history.history['val_dice_coefficient'], label='Val Dice Coeff')
    plt.title('Dice Coefficient History')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    plt.show()

    # --- Inference (on a sample from validation set) ---
    print("\n--- Performing Inference on a sample from the validation set ---")
    if val_patient_dirs:
        # Pick one patient from the validation set for visual demonstration
        demo_patient_dir = val_patient_dirs[0]
        patient_id_folder = os.path.basename(demo_patient_dir)
        print(f"Processing demo patient: {patient_id_folder}")

        demo_mri_processed, demo_mask_gt, original_affine, original_header = \
            preprocess_patient_data(demo_patient_dir, IMG_DIM, MODALITIES_TO_USE, MASK_FILE_SUFFIX)

        if demo_mri_processed is not None:
            X_demo = np.expand_dims(demo_mri_processed, axis=0) # Add batch dimension for prediction
            predicted_mask_probs = model.predict(X_demo)[0] # Predict, remove batch dim
            predicted_mask = (predicted_mask_probs > 0.5).astype(np.uint8) # Threshold probabilities

            # Calculate Dice for this demo prediction
            true_mask_flat = tf.cast(tf.flatten(demo_mask_gt), tf.float32)
            pred_mask_flat = tf.cast(tf.flatten(predicted_mask), tf.float32)
            demo_dice = dice_coefficient(true_mask_flat, pred_mask_flat).numpy()
            print(f"Dice Coefficient for demo patient '{patient_id_folder}': {demo_dice:.4f}")

            # Save the predicted mask
            output_prediction_file = f"{patient_id_folder}_predicted_seg.nii.gz"
            output_prediction_path = os.path.join(PREDICTED_MASKS_DIR, output_prediction_file)
            save_nifti(np.squeeze(predicted_mask), original_affine, original_header, output_prediction_path)

            # Visualize results for the demo patient
            display_overlay(demo_mri_processed, demo_mask_gt, IMG_DIM[2] // 2, axis=2, title="Ground Truth Segmentation", modality_channel=1)
            display_overlay(demo_mri_processed, predicted_mask, IMG_DIM[2] // 2, axis=2, title="Predicted Segmentation", modality_channel=1)
        else:
            print(f"Failed to preprocess demo patient '{patient_id_folder}'.")
    else:
        print("No validation data available for inference demonstration.")

    print("--- Training and Inference Completed ---")

if __name__ == "__main__":
    main()