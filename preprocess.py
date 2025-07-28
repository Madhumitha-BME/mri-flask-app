import nibabel as nib
import numpy as np
import cv2
import os

# Set your input/output directories
DATA_DIR = "data"            # Folder with BraTS folders inside
SAVE_DIR = "processed"       # Will store processed .npy slices
TARGET_SIZE = (128, 128)     # Resize dimensions (H, W)

os.makedirs(SAVE_DIR, exist_ok=True)

def normalize(img):
    """Z-score normalization"""
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std if std > 0 else img

def preprocess_volume(flair_path, seg_path, out_folder):
    print(f"Processing: {flair_path}")
    flair = nib.load(flair_path).get_fdata()
    mask = nib.load(seg_path).get_fdata()

    print(f"Image shape: {flair.shape}, Mask shape: {mask.shape}")

    for i in range(flair.shape[2]):
        img_slice = flair[:, :, i]
        mask_slice = mask[:, :, i]

        # Only save slices with tumor
        if np.max(mask_slice) == 0:
            continue

        img_slice = normalize(img_slice)
        img_resized = cv2.resize(img_slice, TARGET_SIZE)
        mask_resized = cv2.resize(mask_slice, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        os.makedirs(out_folder, exist_ok=True)
        np.save(os.path.join(out_folder, f"img_{i}.npy"), img_resized)
        np.save(os.path.join(out_folder, f"mask_{i}.npy"), mask_resized)

def find_file(folder, keyword):
    """Find a file in folder that contains keyword and ends with .nii or .nii.gz"""
    for file in os.listdir(folder):
        if keyword in file.lower() and file.lower().endswith((".nii", ".nii.gz")):
            return os.path.join(folder, file)
    return None

if __name__ == "__main__":
    for patient_folder in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, patient_folder)
        if not os.path.isdir(full_path):
            continue

        flair_file = find_file(full_path, "flair")
        seg_file = find_file(full_path, "seg")

        if flair_file and seg_file:
            out_dir = os.path.join(SAVE_DIR, patient_folder)
            preprocess_volume(flair_file, seg_file, out_dir)
        else:
            print(f"Skipping {patient_folder} — missing flair or seg file.")