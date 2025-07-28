# src/models.py

from tensorflow import keras
from tensorflow.keras import layers

def unet_3d(input_shape, num_classes):
    """
    Defines a 3D U-Net model for segmentation.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (128, 128, 128, 4)).
        num_classes (int): Number of output classes (1 for binary segmentation).

    Returns:
        keras.Model: The compiled Keras U-Net model.
    """
    inputs = keras.Input(input_shape)

    # Encoder
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up6 = layers.concatenate([layers.Conv3DTranspose(128, 2, strides=(2, 2, 2), padding='same')(conv4), conv3], axis=-1)
    conv6 = layers.Conv3D(128, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same')(conv6), conv2], axis=-1)
    conv7 = layers.Conv3D(64, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv7)

    up8 = layers.concatenate([layers.Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='same')(conv7), conv1], axis=-1)
    conv8 = layers.Conv3D(32, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv8)

    # Output layer: sigmoid for binary segmentation (probabilities between 0 and 1)
    outputs = layers.Conv3D(num_classes, 1, activation='sigmoid', padding='same')(conv8)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model