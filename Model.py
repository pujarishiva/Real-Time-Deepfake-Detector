import cv2          # ?? Needed for image resizing
import numpy as np  # ?? Needed for array operations
import tensorflow as tf  # ?? Needed to load your AI model

# Load your trained model
model = tf.keras.models.load_model("deepfake_model.h5")

def predict_deepfake(face_img):
    """
    Input: face_img = cropped face from video (BGR image)
    Output: score between 0 and 1
    """
    face_img = cv2.resize(face_img, (224,224))  # Resize to model input
    face_img = face_img / 255.0                 # Normalize
    face_img = np.expand_dims(face_img, axis=0)

    prediction = model.predict(face_img)[0][0]
    return prediction