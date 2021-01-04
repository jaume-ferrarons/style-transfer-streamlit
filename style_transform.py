import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Load TF-Hub module.
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image(image_path, image_size=(256, 256)):
    """Loads and preprocesses images."""
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def style_transform(content_image_path, style_image_path):
    start = time.time()
    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

    #output_image_size = 384  # @param {type:"integer"}
    output_image_size = 512  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content_image_path, content_img_size)
    style_image = load_image(style_image_path, style_img_size)

    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]
    print(f"Time enlapsed: {time.time() - start}")
    return stylized_image.numpy().reshape((output_image_size, output_image_size, 3))

if __name__ == "__main__":
    res = style_transform("images/content/cat1.jpg", "images/styles/graffiti.jpg")
    plt.imshow(res)
    plt.show()