# app.py

import os
import numpy as np
import tensorflow as tf
import base64 # Import base64 for decoding
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify # Import jsonify
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

model = None

def load_model_on_startup():
    global model
    model_path = os.path.join(OUTPUT_MODEL_DIR, "brain_mets_unet_multi_patient_model.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please check your training script and paths.")
        return
    custom_objects = {'dice_coefficient': dice_coefficient, 'combined_loss': combined_loss}
    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_model_on_startup()

# --- New function to generate and save a 2D visualization ---
def generate_prediction_image(mri_volume, predicted_mask, filename, slice_idx=None):
    if mri_volume is None or predicted_mask is None:
        return None
    if slice_idx is None:
        slice_idx = mri_volume.shape[2] // 2
    
    mri_slice = mri_volume[:, :, slice_idx, 1]  # Displaying the T1ce modality (channel 1)
    mask_slice = predicted_mask[:, :, slice_idx, 0] 
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mri_slice.T, cmap='gray', origin='lower')
    mask_colored = np.zeros((*mask_slice.shape, 4))
    mask_colored[mask_slice > 0.5] = [1, 0, 0, 0.5]
    ax.imshow(mask_colored, origin='lower')
    ax.set_title("Predicted Segmentation")
    ax.axis('off')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path

# --- Preprocessing function remains the same, but imports its components ---
def preprocess_uploaded_data(uploaded_files_paths, target_img_dim, modalities_to_use):
    # uploaded_files_paths is now a dictionary of {filename: filepath}
    image_modalities_data = {}
    original_affine = None
    original_header = None
    
    for filename, filepath in uploaded_files_paths.items():
        try:
            # nibabel.load() now takes a file path from the temporary disk storage
            data, affine, header = nib.load(filepath).get_fdata(), nib.load(filepath).affine, nib.load(filepath).header
            
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
        if 'files' not in data:
            return jsonify({'message': 'No files data in JSON payload'}), 400

        encoded_files = data['files']
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

            X_new = np.expand_dims(preprocessed_mri, axis=0)

            print(f"Performing prediction on input shape: {X_new.shape}")
            predicted_mask_probs = model.predict(X_new)[0]
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
            # Clean up temporary directory in case of error
            for f_path in uploaded_files_paths.values():
                try: os.remove(f_path)
                except OSError: pass
            try: os.rmdir(request_dir)
            except OSError: pass

            print(f"An error occurred during prediction: {e}")
            return jsonify({'message': f"An error occurred during prediction: {e}"}), 500

    # For GET requests, render the HTML form
    return render_template('index.html')


# --- 4. Provide the endpoint to download files (remains the same) ---
@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)