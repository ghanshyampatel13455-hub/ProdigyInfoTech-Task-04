import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/hand_gesture_model.keras")

# Class names
classes = {
    0: "Palm",
    1: "L",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumb",
    5: "Index",
    6: "OK",
    7: "Palm Moved",
    8: "C",
    9: "Down"
}

# Enter image path
image_path = input("Enter image path: ")

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# Preprocess
img_resized = cv2.resize(img, (64, 64))
img_resized = img_resized / 255.0
img_resized = np.expand_dims(img_resized, axis=0)

# Prediction
prediction = model.predict(img_resized)

class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\nPrediction:", classes[class_index])
print("Confidence: {:.2f}%".format(confidence))