# app.py

import os
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import uuid
import nibabel as nib
import matplotlib.pyplot as plt

# Import functions from your src modules
from src.data_processor import save_nifti, normalize_intensity, pad_or_crop_volume
from src.metrics import dice_coefficient, combined_loss

# Import configuration parameters
from config import (
    IMG_DIM, MODALITIES_TO_USE, MASK_FILE_SUFFIX, PREDICTED_MASKS_DIR,
    OUTPUT_MODEL_DIR, UPLOAD_FOLDER
)

# --- 1. Set up the Flask App ---
app = Flask(__name__)
# Max content length might still be useful for total JSON payload size
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024 # 1 GB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Global model variable
model = None 

def load_model_on_startup():
    global model
    # --- CHANGE: Load the original .h5 model ---
    model_path = os.path.join(OUTPUT_MODEL_DIR, "brain_mets_unet_multi_patient_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Original .h5 model not found at {model_path}.")
        print("Please ensure you have trained your model (run train.py) and the .h5 file exists.")
        return

    print(f"Loading original .h5 model from: {model_path}")
    try:
        # Load the Keras model with custom objects
        # These custom objects are essential for loading models that use custom loss/metrics
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'combined_loss': combined_loss
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Original .h5 model loaded successfully.")
    except Exception as e:
        print(f"Error loading original .h5 model: {e}")
        # Consider exiting or raising an error if model loading is critical
        
# Call the model loading function immediately after app creation
load_model_on_startup()


# --- New function to generate and save a 2D visualization ---
def generate_prediction_image(mri_volume, predicted_mask, filename, slice_idx=None):
    """
    Creates a PNG visualization of a central MRI slice with the predicted mask overlaid.
    """
    if mri_volume is None or predicted_mask is None:
        return None

    # Use the central slice if not specified
    if slice_idx is None:
        slice_idx = mri_volume.shape[2] // 2
    
    # Ensure mri_volume and predicted_mask are correctly squeezed before slicing
    mri_volume_squeezed = np.squeeze(mri_volume)
    predicted_mask_squeezed = np.squeeze(predicted_mask)

    mri_slice = mri_volume_squeezed[:, :, slice_idx, 1]  # Displaying the T1ce modality (channel 1)
    mask_slice = predicted_mask_squeezed[:, :, slice_idx] # Mask is already binary after squeeze
    
    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mri_slice.T, cmap='gray', origin='lower')
    
    # Overlay the mask in a semi-transparent color
    mask_colored = np.zeros((*mask_slice.shape, 4))
    mask_colored[mask_slice > 0.5] = [1, 0, 0, 0.5] # Red with 50% transparency
    ax.imshow(mask_colored, origin='lower')
    
    ax.set_title("Predicted Segmentation")
    ax.axis('off')
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the figure to free up memory
    
    return output_path


# --- Preprocessing function remains the same, but imports its components ---
def preprocess_uploaded_data(uploaded_files_paths, target_img_dim, modalities_to_use):
    """
    Preprocesses uploaded NIfTI files by matching them to modalities.
    """
    image_modalities_data = {}
    original_affine = None
    original_header = None
    
    for filename, filepath in uploaded_files_paths.items():
        try:
            # nibabel.load() now takes a file path from the temporary disk storage
            # Ensure data, affine, header are correctly extracted from nibabel load
            nifti_img = nib.load(filepath)
            data = nifti_img.get_fdata()
            affine = nifti_img.affine
            header = nifti_img.header
            
            modality = None
            filename_base = filename.lower().split('.')[0]
            last_part = filename_base.split('_')[-1]
            
            for mod in modalities_to_use:
                if mod == last_part:
                    modality = mod
                    break
            
            if modality:
                if modality not in image_modalities_data:
                    image_modalities_data[modality] = (data, affine, header)
                if original_affine is None:
                    original_affine = affine
                    original_header = header
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None, None

    if len(image_modalities_data) != len(modalities_to_use):
        print(f"Error: Expected {len(modalities_to_use)} modalities, but found only {len(image_modalities_data)}.")
        return None, None, None
        
    preprocessed_images_ordered = []
    for mod in modalities_to_use:
        data, _, _ = image_modalities_data[mod]
        data = normalize_intensity(data, method='minmax')
        preprocessed_images_ordered.append(pad_or_crop_volume(data, target_img_dim))
    
    combined_image = np.stack(preprocessed_images_ordered, axis=-1)
    return combined_image, original_affine, original_header


# --- 3. Handle File Uploads and Prediction Logic (Modified for JSON/Base64) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Expect JSON payload instead of form-data
        if not request.is_json:
            return jsonify({'message': 'Request must be JSON'}), 400

        data = request.get_json()
        encoded_files = data.get('files') # Use .get() for safer access
        if not encoded_files:
            return jsonify({'message': 'No files received.'}), 400

        # Create a temporary directory for this specific request
        request_id = str(uuid.uuid4())
        request_dir = os.path.join(app.config['UPLOAD_FOLDER'], request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        uploaded_files_paths = {} # To store {filename: filepath}
        
        try:
            for encoded_file in encoded_files:
                filename = secure_filename(encoded_file['filename'])
                base64_content = encoded_file['content']
                
                # Decode base64 and save to temporary file
                file_content = base64.b64decode(base64_content)
                file_path = os.path.join(request_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                uploaded_files_paths[filename] = file_path

            if len(uploaded_files_paths) < len(MODALITIES_TO_USE):
                # Clean up temporary directory before returning error
                for f_path in uploaded_files_paths.values(): os.remove(f_path)
                os.rmdir(request_dir)
                return jsonify({'message': 'Please upload all required NIfTI files.'}), 400

            preprocessed_mri, new_affine, new_header = preprocess_uploaded_data(
                uploaded_files_paths, IMG_DIM, MODALITIES_TO_USE
            )
            
            # Clean up temporary directory after preprocessing
            for f_path in uploaded_files_paths.values(): os.remove(f_path)
            os.rmdir(request_dir)

            if preprocessed_mri is None:
                return jsonify({'message': 'Preprocessing failed. Please check your uploaded files are all present and correctly named.'}), 500

            # --- Keras Model Inference ---
            # Input data for Keras model must be float32 and match model's expected input shape
            input_tensor = np.expand_dims(preprocessed_mri, axis=0).astype(np.float32)
            
            # Direct prediction using the Keras model
            predicted_mask_probs = model.predict(input_tensor)[0]
            
            predicted_mask = (predicted_mask_probs > 0.5).astype(np.uint8)

            output_image_filename = f"predicted_mask_{uuid.uuid4()}.png"
            generate_prediction_image(
                preprocessed_mri,
                predicted_mask,
                output_image_filename
            )
            
            download_link = url_for('download_file', filename=output_image_filename)
            return jsonify({'message': 'Prediction successful!', 'download_link': download_link}), 200

        except Exception as e:
            # Error cleanup
            for f_path in uploaded_files_paths.values():
                try: os.remove(f_path)
                except OSError: pass
            try: os.rmdir(request_dir)
            except OSError: pass

            print(f"An error occurred during prediction: {e}")
            return jsonify({'message': f"An error occurred during prediction: {e}"}), 500

    return render_template('index.html')


# --- 4. Provide the endpoint to download files (remains the same) ---
@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)