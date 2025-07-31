# app_render.py

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
app_render = Flask(__name__) 
app_render.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024 # 1 GB
app_render.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app_render.secret_key = 'supersecretkey'

model_render = None 

def load_model_on_startup_render():
    global model_render
    tflite_model_path = os.path.join(OUTPUT_MODEL_DIR, "model_quantized_float16.tflite")
    
    if not os.path.exists(tflite_model_path):
        print(f"Error: Quantized TFLite model not found at {tflite_model_path}.")
        print("Please run 'quantize_model.py' script first to create it.")
        return

    print(f"Loading TFLite model from: {tflite_model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_render = {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details
        }
        print("TFLite model loaded successfully for Render deployment.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        
load_model_on_startup_render()


# --- New function to generate and save a 2D visualization ---
def generate_prediction_image(mri_volume, predicted_mask, filename, slice_idx=None):
    if mri_volume is None or predicted_mask is None:
        return None
    if slice_idx is None:
        slice_idx = mri_volume.shape[2] // 2
    
    mri_volume_squeezed = np.squeeze(mri_volume)
    predicted_mask_squeezed = np.squeeze(predicted_mask)

    mri_slice = mri_volume_squeezed[:, :, slice_idx, 0]  # Assuming we'll send a single channel file, so channel 0
    mask_slice = predicted_mask_squeezed[:, :, slice_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mri_slice.T, cmap='gray', origin='lower')
    mask_colored = np.zeros((*mask_slice.shape, 4))
    mask_colored[mask_slice > 0.5] = [1, 0, 0, 0.5]
    ax.imshow(mask_colored, origin='lower')
    ax.set_title("Predicted Segmentation")
    ax.axis('off')
    
    output_path = os.path.join(app_render.config['UPLOAD_FOLDER'], filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path


# --- Preprocessing function for single file test ---
def preprocess_uploaded_data_single_file(filename, filepath, target_img_dim):
    """
    Preprocesses a single uploaded NIfTI file for the test.
    """
    try:
        nifti_img = nib.load(filepath)
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        header = nifti_img.header
        
        # We assume this single file represents the main modality (e.g., T1ce or T1)
        # We need to make it a 4D tensor with one channel.
        data = normalize_intensity(data, method='minmax')
        data = pad_or_crop_volume(data, target_img_dim)
        combined_image = np.expand_dims(data, axis=-1) # Add channel dim
        
        return combined_image, affine, header
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None, None


# --- 3. Handle File Uploads and Prediction Logic (Modified for Single JSON/Base64) ---
@app_render.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({'message': 'Request must be JSON'}), 400

        data = request.get_json()
        # --- CHANGE: Expect single file object directly ---
        filename = data.get('filename')
        base64_content = data.get('content')
        
        if not filename or not base64_content:
            return jsonify({'message': 'No filename or content in JSON payload.'}), 400

        request_id = str(uuid.uuid4())
        request_dir = os.path.join(app_render.config['UPLOAD_FOLDER'], request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        file_path = os.path.join(request_dir, secure_filename(filename)) # Path for the single file
        
        try:
            # Decode base64 and save to temporary file
            file_content = base64.b64decode(base64_content)
            with open(file_path, 'wb') as f:
                f.write(file_content)

            # --- Use the single file preprocessing function ---
            preprocessed_mri, new_affine, new_header = preprocess_uploaded_data_single_file(
                filename, file_path, IMG_DIM
            )
            
            # Clean up temporary file and directory
            os.remove(file_path)
            os.rmdir(request_dir)

            if preprocessed_mri is None:
                return jsonify({'message': 'Preprocessing failed for the single file.'}), 500

            # --- TFLite Inference ---
            input_tensor = np.expand_dims(preprocessed_mri, axis=0).astype(np.float32)
            
            interpreter = model_render['interpreter']
            input_details = model_render['input_details']
            output_details = model_render['output_details']

            # Make sure input_tensor shape matches model's expected input shape
            # This might require reshaping preprocessed_mri if the model expects multiple channels
            # For this test, we assume the model can handle a single channel or we adapt the model.
            # Usually, you'd stack the one channel into the expected 4-channel input if the model is fixed.
            # Example: if model expects (D,H,W,4) and preprocessed_mri is (D,H,W,1), you need to pad channels.
            # For now, let's assume the TFLite model derived from a multi-channel Keras model will still expect 4 channels.
            # If so, you'd need to create a 4-channel dummy input.
            # THIS IS A CRITICAL ASSUMPTION FOR THE TEST:
            # If model expects 4 channels, but we give it 1 channel, it will fail during inference.
            # So, let's create a 4-channel input by padding the single channel.
            expected_input_shape = input_details[0]['shape'] # e.g., (1, 128, 128, 128, 4)
            if preprocessed_mri.shape[-1] != expected_input_shape[-1]:
                # If uploaded file is 1-channel, but model expects 4, pad with zeros
                padded_input = np.zeros(expected_input_shape[1:], dtype=np.float32) # Shape (D,H,W,C)
                padded_input[:,:,:,0] = np.squeeze(preprocessed_mri) # Put the single channel into the first one
                input_tensor = np.expand_dims(padded_input, axis=0).astype(np.float32)
            else:
                input_tensor = np.expand_dims(preprocessed_mri, axis=0).astype(np.float32)


            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            
            predicted_mask_probs = interpreter.get_tensor(output_details[0]['index'])[0] 
            predicted_mask = (predicted_mask_probs > 0.5).astype(np.uint8)

            output_image_filename = f"predicted_mask_{uuid.uuid4()}.png"
            generate_prediction_image(
                preprocessed_mri, # Pass the original preprocessed_mri for visualization context
                predicted_mask,
                output_image_filename
            )
            
            download_link = url_for('download_file', filename=output_image_filename)
            return jsonify({'message': 'Prediction successful!', 'download_link': download_link}), 200

        except Exception as e:
            # Error cleanup
            if 'file_path' in locals() and os.path.exists(file_path): os.remove(file_path)
            if 'request_dir' in locals() and os.path.exists(request_dir): os.rmdir(request_dir)

            print(f"An error occurred during prediction: {e}")
            return jsonify({'message': f"An error occurred during prediction: {e}"}), 500

    return render_template('index.html')


# --- 4. Provide the endpoint to download files ---
@app_render.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app_render.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app_render.run(debug=True)