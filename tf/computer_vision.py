import tensorflow as tf
import math
import tensorflow as tf

def random_crop_resize(image, cropped_factor=0.95, seed=None):
    """
    Randomly crops an image and resizes it back to its original dimensions.

    Args:
        image: A 3D tensor representing an image (H, W, C).
        cropped_factor: The factor by which to crop the image (default: 0.95).
        seed: Optional integer seed for reproducibility (default: None).

    Returns:
        The cropped and resized image.
    """
    if cropped_factor <= 0 or cropped_factor > 1.0:
        raise ValueError("cropped_factor must be between 0 (exclusive) and 1.0 (inclusive)")

    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    crop_h = tf.cast(tf.cast(h, tf.float32) * cropped_factor, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * cropped_factor, tf.int32)

    image = tf.image.random_crop(image, size=(crop_h, crop_w, 3), seed=seed)

    image = tf.image.resize(image, size=(h, w))
    return image


def random_erasing(image, prob=0.5, min_area=0.02, max_area=0.3,
                   min_ratio=0.3, max_ratio=3.0, mode="black", seed=None):
    """
    Randomly erases a rectangular region of an image (Zhong et al., 2020).

    Args:
        image:      A 3D tensor (H, W, C) with pixel values in [0, 255].
        prob:       Probability of applying the augmentation (default: 0.5).
        min_area:   Minimum erased area as a fraction of the image (default: 0.02).
        max_area:   Maximum erased area as a fraction of the image (default: 0.3).
        min_ratio:  Minimum aspect ratio (h/w) of the erased region (default: 0.3).
        max_ratio:  Maximum aspect ratio (h/w) of the erased region (default: 3.0).
        mode:       Fill value — "black" (0), "white" (255), or "noise" (default: "black").
        seed:       Optional integer seed for reproducibility. When set, uses stateless
                    ops with split seeds so each random draw is independent (default: None).

    Returns:
        The image with a randomly erased rectangle, or the original image unchanged.

    Raises:
        (no explicit raises — invalid mode silently falls through to "noise" behaviour)

    References:
        Zhong et al., "Random Erasing Data Augmentation", AAAI 2020.
        https://arxiv.org/abs/1708.04896
    """

    img_h = tf.shape(image)[0]
    img_w = tf.shape(image)[1]
    img_area = tf.cast(img_h * img_w, tf.float32)

    if mode == "black":
        noise_min, noise_max = 0.0, 0.0
    elif mode == "white":
        noise_min, noise_max = 255.0, 255.0
    else:
        noise_min, noise_max = 0.0, 255.0

    # Split one seed into N independent sub-seeds
    if seed is not None:
        seeds = tf.random.experimental.stateless_split([seed, seed + 1], num=6)
        # seeds[i] is a (2,) tensor usable by stateless ops
    
    def _uniform(i, *args, **kwargs):
        """Dispatch to stateless or stateful uniform depending on seed."""
        if seed is not None:
            return tf.random.stateless_uniform(*args, seed=seeds[i], **kwargs)
        return tf.random.uniform(*args, **kwargs)

    def erase(image):
        erase_area  = _uniform(0, [], min_area, max_area) * img_area
        aspect_ratio = _uniform(1, [], min_ratio, max_ratio)

        h = tf.minimum(tf.cast(tf.math.sqrt(erase_area * aspect_ratio), tf.int32), img_h - 1)
        w = tf.minimum(tf.cast(tf.math.sqrt(erase_area / aspect_ratio), tf.int32), img_w - 1)
        h = tf.maximum(h, 1)
        w = tf.maximum(w, 1)

        top  = _uniform(2, [], 0, img_h - h, dtype=tf.int32)
        left = _uniform(3, [], 0, img_w - w, dtype=tf.int32)
        noise = _uniform(4, [h, w, 3], noise_min, noise_max)

        padded_noise = tf.pad(noise,        [[top, img_h-top-h], [left, img_w-left-w], [0,0]])
        padded_mask  = tf.pad(tf.ones([h, w, 3]), [[top, img_h-top-h], [left, img_w-left-w], [0,0]])

        padded_noise = tf.ensure_shape(padded_noise, image.shape)
        padded_mask  = tf.ensure_shape(padded_mask,  image.shape)

        return image * (1 - padded_mask) + padded_noise * padded_mask

    return tf.cond(
        _uniform(5, []) < prob,
        true_fn=lambda: erase(image),
        false_fn=lambda: image
    )



def _rotate_with_transform(image, angle):
    """Applies a rotation using TF's projective transform."""
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    cx, cy = w / 2.0, h / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    # Projective transform matrix (8 coefficients, row-major)
    # Maps output pixel (x', y') back to input pixel (x, y)
    transform = [
        cos_a,  sin_a, cx - cx * cos_a - cy * sin_a,
       -sin_a,  cos_a, cy + cx * sin_a - cy * cos_a,
        0.0,    0.0
    ]
    transform = tf.reshape(tf.stack(transform), [1, 8])

    img_4d    = tf.expand_dims(image, 0)                    # (1, H, W, C)
    output_shape = tf.stack([h, w])

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(img_4d, tf.float32),
        transforms=transform,
        output_shape=tf.cast(output_shape, tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0
    )
    return tf.squeeze(rotated, axis=0)                      # (H, W, C)


def random_rotation(image, max_angle=30, mode="black", seed=None):
    """
    Randomly rotates an image and fills empty pixels with black, white, or noise.

    Args:
        image:     A 3D tensor (H, W, C) with pixel values in [0, 255].
        max_angle: Maximum rotation angle in degrees (default: 30).
        mode:      Fill for empty pixels — "black" (0), "white" (255),
                   or "noise" (default: "black").
        seed:      Optional integer seed for reproducibility (default: None).

    Returns:
        The rotated image with background filled according to mode.

    Raises:
        ValueError: If mode is not one of "black", "white", or "noise".
    """
    if mode not in ("black", "white", "noise"):
        raise ValueError(f"mode must be 'black', 'white', or 'noise', got '{mode}'")

    if seed is not None:
        seeds = tf.random.experimental.stateless_split([seed, seed + 1], num=2)

    def _uniform(i, *args, **kwargs):
        if seed is not None:
            return tf.random.stateless_uniform(*args, seed=seeds[i], **kwargs)
        return tf.random.uniform(*args, **kwargs)

    max_rad = max_angle * (math.pi / 180.0)
    angle = _uniform(0, [], -max_rad, max_rad)

    rotated = _rotate_with_transform(image, angle)

    if mode == "black":
        return rotated

    # Rotate an all-ones canvas to find which pixels are empty
    ones = tf.ones_like(image, dtype=tf.float32)
    valid_mask = _rotate_with_transform(ones, angle)

    if mode == "white":
        fill = tf.ones_lik