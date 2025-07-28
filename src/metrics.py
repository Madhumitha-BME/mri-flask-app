# src/metrics.py

import tensorflow as tf
from tensorflow import keras

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for segmentation evaluation.
    Args:
        y_true (tf.Tensor): Ground truth masks (binary).
        y_pred (tf.Tensor): Predicted probabilities from the model.
        smooth (float): Smoothing factor to prevent division by zero.
    Returns:
        tf.Tensor: Dice coefficient score.
    """
    # --- CHANGE THESE TWO LINES ---
    # Replace tf.flatten with tf.reshape(..., -1) to flatten the tensor
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32) # Predictions are probabilities

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function (1 - Dice coefficient)."""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combines Binary Cross-Entropy (BCE) loss and Dice loss.
    Often provides better stability and performance for segmentation tasks with class imbalance.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice