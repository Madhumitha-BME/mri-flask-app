# src/data_processor.py

import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
import os

def load_nifti_to_numpy(file_path):
    """Loads a NIfTI file and returns the image data, affine matrix, and header."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img.affine, img.header
    except Exception as e:
        print(f"Error loading NIfTI file {file_path}: {e}")
        return None, None, None

def save_nifti(data, affine, header, output_path):
    """Saves a NumPy array back to a NIfTI file."""
    # Ensure data type is float32, which is common for NIfTI images
    if data.dtype != np.float32 and data.dtype != np.uint8:
        print(f"Warning: Converting data from {data.dtype} to float32 for NIfTI save.")
        data = data.astype(np.float32)

    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, output_path)
    print(f"Data saved to: {output_path}")

def normalize_intensity(image_data, method='minmax'):
    """Normalizes image intensities."""
    image_data = image_data.astype(np.float32)
    if method == 'minmax':
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if (max_val - min_val) > 0:
            return (image_data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image_data)
    elif method == 'zscore':
        mean_val = np.mean(image_data)
        std_val = np.std(image_data)
        if std_val > 0:
            return (image_data - mean_val) / std_val
        else:
            return np.zeros_like(image_data)
    return image_data

def resample_volume(volume, original_spacing, target_spacing, is_mask=False):
    """
    Resamples a 3D volume to a target spacing.
    Uses order=3 for images (cubic) and order=0 for masks (nearest neighbor).
    """
    resize_factor = [s_orig / s_target for s_orig, s_target in zip(original_spacing, target_spacing)]
    if is_mask:
        return zoom(volume, resize_factor, order=0) # Nearest neighbor for masks
    else:
        return zoom(volume, resize_factor, order=3) # Cubic for images

def pad_or_crop_volume(volume, target_shape):
    """Pads or crops a 3D volume to target_shape, centering the content."""
    current_shape = volume.shape
    padded_volume = np.zeros(target_shape, dtype=volume.dtype)

    # Calculate slices for the original volume
    start_old = [max(0, (current_shape[i] - target_shape[i]) // 2) for i in range(3)]
    end_old = [start_old[i] + min(current_shape[i], target_shape[i]) for i in range(3)]
    slices_old = tuple(slice(start_old[i], end_old[i]) for i in range(3))

    # Calculate slices for the new padded volume
    start_new = [max(0, (target_shape[i] - current_shape[i]) // 2) for i in range(3)]
    end_new = [start_new[i] + min(current_shape[i], target_shape[i]) for i in range(3)]
    slices_new = tuple(slice(start_new[i], end_new[i]) for i in range(3))

    padded_volume[slices_new] = volume[slices_old]
    return padded_volume

def preprocess_patient_data(patient_dir, target_img_dim, modalities_to_load, mask_suffix):
    """
    Loads, preprocesses, and concatenates multi-modal MRI data for a single patient.
    Expects all relevant NIfTI files to be directly in `patient_dir`.
    """
    image_modalities_data = []
    original_affine = None
    original_header = None
    original_spacing = None

    patient_id = os.path.basename(patient_dir)

    # Load and preprocess image modalities
    for mod in modalities_to_load:
        img_path = os.path.join(patient_dir, f"{patient_id}_{mod}.nii")
        data, affine, header = load_nifti_to_numpy(img_path)

        if data is None:
            print(f"Error: Missing or corrupted file for {patient_id} modality {mod} at {img_path}. Cannot proceed.")
            return None, None, None, None # Return None for all if any critical file is missing

        if original_affine is None: # Store affine/header from the first loaded image
            original_affine = affine
            original_header = header
            original_spacing = header.get_zooms()[:3]

        # Optional: Resample to a common isotropic spacing if needed
        # data = resample_volume(data, original_spacing, TARGET_VOXEL_SPACING, is_mask=False)

        data = normalize_intensity(data, method='minmax') # Normalize each modality
        image_modalities_data.append(data)

    # Load and preprocess mask
    mask_path = os.path.join(patient_dir, f"{patient_id}{mask_suffix}.nii")
    mask_data, _, _ = load_nifti_to_numpy(mask_path)
    if mask_data is None:
        print(f"Error: Missing or corrupted mask for {patient_id} at {mask_path}. Cannot proceed.")
        return None, None, None, None

    # Optional: Resample mask using nearest neighbor
    # mask_data = resample_volume(mask_data, original_spacing, TARGET_VOXEL_SPACING, is_mask=True)

    mask_data = (mask_data > 0).astype(np.float32) # Ensure binary mask (0 or 1)

    # Pad or crop all volumes to target_img_dim
    preprocessed_images = []
    for img_data in image_modalities_data:
        preprocessed_images.append(pad_or_crop_volume(img_data, target_img_dim))

    preprocessed_mask = pad_or_crop_volume(mask_data, target_img_dim)

    # Stack modalities as channels
    combined_image = np.stack(preprocessed_images, axis=-1) # Shape: (D, H, W, Channels)
    combined_mask = np.expand_dims(preprocessed_mask, axis=-1) # Add channel dim for mask (D, H, W, 1)

    return combined_image, combined_mask, original_affine, original_header

def data_generator(patient_dirs, target_img_dim, batch_size, modalities, mask_suffix):
    """
    Generates batches of preprocessed MRI images and masks from a list of patient directories.
    This is for training with multiple patients.
    """
    num_patients = len(patient_dirs)
    while True:
        np.random.shuffle(patient_dirs) # Shuffle patient order for each epoch
        for i in range(0, num_patients, batch_size):
            batch_patient_dirs = patient_dirs[i : i + batch_size]
            batch_images = []
            batch_masks = []
            for p_dir in batch_patient_dirs:
                img, mask, _, _ = preprocess_patient_data(p_dir, target_img_dim, modalities, mask_suffix)
                if img is not None and mask is not None:
                    batch_images.append(img)
                    batch_masks.append(mask)
            if batch_images and batch_masks:
                yield np.array(batch_images), np.array(batch_masks)