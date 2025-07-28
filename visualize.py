import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
# Change this to any patient folder inside the processed directory
patient_folder = "processed/BraTS20_Training_032"

# You can also list all available slices
available_slices = sorted([f for f in os.listdir(patient_folder) if f.startswith("img_")])
print("Available slices:", available_slices)

# Pick a slice number that exists (e.g., from the printed list)
image_index = 111  # Change this to match a real slice number

# Construct full paths
img_path = os.path.join(patient_folder, f"img_{image_index}.npy")
mask_path = os.path.join(patient_folder, f"mask_{image_index}.npy")

# Check if files exist
if not os.path.exists(img_path) or not os.path.exists(mask_path):
    print(f"Error: Slice img_{image_index}.npy or its mask not found in {patient_folder}")
    exit()

# Load the image and mask
img = np.load(img_path)
mask = np.load(mask_path)

# === PLOT ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("FLAIR Slice")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
