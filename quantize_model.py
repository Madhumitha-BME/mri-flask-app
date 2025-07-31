# quantize_model.py (Run this script locally)

import tensorflow as tf
import os

from src.metrics import dice_coefficient, combined_loss

# Path to your existing trained model
model_path = 'trained_models/brain_mets_unet_multi_patient_model.h5'
output_tflite_path = 'trained_models/model_quantized_float16.tflite'

os.makedirs('trained_models', exist_ok=True)

print(f"Loading Keras model from: {model_path}")
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'dice_coefficient': dice_coefficient,
            'combined_loss': combined_loss
        }
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Ensure src/metrics.py is in the Python path.")
    exit()

print("Starting float16 quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS    # Enable ops that need full TF kernels (like MaxPool3D)
]

try:
    tflite_model = converter.convert()
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Float16 quantized model saved to: {output_tflite_path}")
    print(f"Original model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"Quantized model size: {os.path.getsize(output_tflite_path) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"Error during model conversion: {e}")