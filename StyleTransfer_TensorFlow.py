import tensorflow as tf
import numpy as np


def transfer_style(content_image_file_name, style_image_file_name):
    content_image = load_image_not_cropped(f"slika osebe/{content_image_file_name}")
    style_image = load_style_image(f"razglednica/{style_image_file_name}")
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

    # Load model locally
    local_location = "magenta_arbitrary-image-stylization-v1-256_2"
    module = tf.saved_model.load(local_location, tags=None, options=None)

    # Save original shape
    shape = [content_image.shape[1], content_image.shape[2]]

    # Transfer style
    outputs = module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Resize back to normal
    stylized_image = tf.image.resize(stylized_image, shape)

    # Convert to normal image
    stylized_image = tf.squeeze(stylized_image, axis=[0], name=None)
    stylized_image = tf.image.convert_image_dtype(stylized_image, dtype=np.uint8, saturate=True, name=None)

    return stylized_image


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_style_image(image_url):
    """Loads and preprocesses style image."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = tf.io.decode_image(tf.io.read_file(image_url), channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    # Resize style image to (256, 256), the model needs this size, leave it!
    img = tf.image.resize(img, (256, 256), preserve_aspect_ratio=True)
    return img


def load_image_not_cropped(image_url):
    """Loads and preprocesses image."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    return tf.io.decode_image(tf.io.read_file(image_url), channels=3, dtype=tf.float32)[tf.newaxis, ...]
