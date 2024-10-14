# helpers/predictor.py

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the model once when the module is imported
model = keras.models.load_model("model_correct.h5")

def predict(img_path):
    try:
        # Load the Image
        img = Image.open(img_path)
        
        # Resize Image to size of (300, 300)
        img = img.resize((300, 300))
        
        # Convert Image to a numpy array
        img_array = image.img_to_array(img, dtype=np.uint8)
        
        # Scaling the Image Array values between 0 and 1
        img_array = img_array / 255.0
        
        # Expand dimensions to match the model's expected input
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 300, 300, 3)
        
        # Get the Predicted Label for the loaded Image
        prediction = model.predict(img_array)
        
        # Label array
        labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
        
        # Predicted Class
        predicted_class = labels[np.argmax(prediction[0], axis=-1)]
        
        return [prediction, predicted_class]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [None, "Prediction Failed"]

if __name__ == "__main__":
    # Test the predict function with a sample image
    sample_image_path = "path_to_sample_image.jpg"  # Replace with actual image path
    if os.path.exists(sample_image_path):
        prediction, predicted_class = predict(sample_image_path)
        print(f"Predicted Class: {predicted_class}")
        print(f"Prediction Probabilities: {prediction}")
    else:
        print(f"Sample image not found at {sample_image_path}. Please provide a valid image path.")
