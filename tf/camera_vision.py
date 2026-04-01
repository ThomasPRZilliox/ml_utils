import tensorflow as tf

def random_crop_resize(image, cropped_factor=0.95):
    """
    Randomly crops an image and resizes it back to its original dimensions.
    
    Args:
        image: A 3D tensor representing an image (H, W, C).
        cropped_factor: The factor by which to crop the image (default: 0.95).
        
    Returns:
        The cropped and resized image.
    """
    shape = tf.shape(image)          # dynamic shape — works inside tf.data/tf.function
    h, w = shape[0], shape[1]

    crop_h = tf.cast(tf.cast(h, tf.float32) * cropped_factor, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * cropped_factor, tf.int32)

    image = tf.image.random_crop(image, size=(crop_h, crop_w, 3))

    # resize back to original dimensions (also dynamic)
    image = tf.image.resize(image, size=(h, w))
    return image
