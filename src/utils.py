# src/utils.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def display_overlay(mri_data, mask_data, slice_idx, axis=2, alpha=0.5, title="MRI with Segmentation Overlay", modality_channel=0):
    """
    Displays an MRI slice with an overlay of the segmentation mask.
    """
    if mri_data is None or mask_data is None:
        return
        
    # --- FIX 1: SQUEEZE THE MASK DATA FIRST ---
    mask_data = np.squeeze(mask_data)
    
    if mri_data.ndim == 4 and modality_channel < mri_data.shape[-1]:
        mri_slice = mri_data[:, :, :, modality_channel]
    elif mri_data.ndim == 3:
        mri_slice = mri_data
    else:
        print(f"Warning: Cannot display modality channel {modality_channel}. Defaulting to first channel.")
        mri_slice = mri_data[:, :, :, 0]

    if axis == 0:
        mri_slice_data = mri_slice[slice_idx, :, :]
        mask_slice = mask_data[slice_idx, :, :]
    elif axis == 1:
        mri_slice_data = mri_slice[:, slice_idx, :]
        mask_slice = mask_data[:, slice_idx, :]
    elif axis == 2:
        mri_slice_data = mri_slice[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
    else:
        print("Invalid axis. Choose 0, 1, or 2.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(mri_slice_data.T, cmap='gray', origin='lower')
    ax[0].set_title(f"Original MRI Slice {slice_idx} (Channel {modality_channel})")
    ax[0].axis('off')

    ax[1].imshow(mri_slice_data.T, cmap='gray', origin='lower')
    mask_colored = np.zeros((*mask_slice.shape, 4))
    mask_colored[mask_slice > 0.5] = [1, 0, 0, alpha]
    ax[1].imshow(mask_colored, origin='lower')
    ax[1].set_title(f"{title} (Slice {slice_idx})")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

def display_comparative_overlay(mri_data, gt_mask, pred_mask, slice_idx, axis=2, title="Comparative Segmentation Overlay", modality_channel=0, alpha=0.5):
    """
    Displays an MRI slice with a comparative overlay of both ground truth and predicted masks.
    """
    if mri_data is None or gt_mask is None or pred_mask is None:
        return

    # --- FIX 2: SQUEEZE BOTH MASKS FIRST ---
    gt_mask = np.squeeze(gt_mask)
    pred_mask = np.squeeze(pred_mask)

    if mri_data.ndim == 4 and modality_channel < mri_data.shape[-1]:
        mri_slice = mri_data[:, :, :, modality_channel]
    elif mri_data.ndim == 3:
        mri_slice = mri_data
    else:
        print(f"Warning: Cannot display modality channel {modality_channel}. Defaulting to first channel.")
        mri_slice = mri_data[:, :, :, 0]

    if axis == 0:
        mri_slice_data = mri_slice[slice_idx, :, :]
        gt_mask_slice = gt_mask[slice_idx, :, :]
        pred_mask_slice = pred_mask[slice_idx, :, :]
    elif axis == 1:
        mri_slice_data = mri_slice[:, slice_idx, :]
        gt_mask_slice = gt_mask[:, slice_idx, :]
        pred_mask_slice = pred_mask[:, slice_idx, :]
    elif axis == 2:
        mri_slice_data = mri_slice[:, :, slice_idx]
        gt_mask_slice = gt_mask[:, :, slice_idx]
        pred_mask_slice = pred_slice[:, :, slice_idx]
    else:
        print("Invalid axis. Choose 0, 1, or 2.")
        return

    combined_overlay = np.zeros((*gt_mask_slice.shape, 4))
    
    combined_overlay[gt_mask_slice > 0.5, 1] = 1
    combined_overlay[gt_mask_slice > 0.5, 3] = alpha

    combined_overlay[pred_mask_slice > 0.5, 0] = 1
    combined_overlay[pred_mask_slice > 0.5, 3] = alpha
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mri_slice_data.T, cmap='gray', origin='lower')
    plt.imshow(combined_overlay, origin='lower')

    plt.title(title)
    plt.axis('off')
    plt.show()

def display_difference(mri_slice, gt_slice, pred_slice, title="Segmentation Difference"):
    """
    Displays the difference between the ground truth and predicted masks.
    """
    # --- FIX 3: SQUEEZE THE MASKS FIRST ---
    gt_slice = np.squeeze(gt_slice)
    pred_slice = np.squeeze(pred_slice)

    difference_image = np.zeros((*mri_slice.shape, 3), dtype=np.float32)

    difference_image[gt_slice > 0.5] = [1, 1, 1]
    difference_image[(gt_slice > 0.5) & (pred_slice < 0.5)] = [0, 1, 0]
    difference_image[(gt_slice < 0.5) & (pred_slice > 0.5)] = [1, 0, 0]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[0].set_title("Original MRI Slice")
    ax[0].axis('off')

    ax[1].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[1].imshow(difference_image, alpha=0.5, origin='lower')
    ax[1].set_title(title)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()