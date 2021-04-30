import cv2
import numpy as np
import tensorflow as tf

def force_from_image_numpy(image_np, estimation_model, input_size=(460, 680)):
    # resize the finger image to be readble by the estimation model
    img_cropped_resized = cv2.resize(image_np, input_size)
    image = np.asarray(img_cropped_resized, dtype=np.float32)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    forces = estimation_model(input_tensor)
    return forces.numpy()[0]


def save_force_text(forces, path_to_save):
    # save the forces in a txt file with the frame numbers

    return 0

def force_print_image_numpy(forces, image_np):
    # save the forces in a txt file with the frame numbers

    return 0
