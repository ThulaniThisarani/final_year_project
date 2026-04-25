"""
Single Image Prediction Script
Cinnamon Leaf Disease Classification

Usage:
  python predict_cinnamon_leaf.py path/to/image.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# =========================
# 1) Config
# =========================
MODEL_PATH = "best_cinnamon_model.keras"  
# or "cinnamon_disease_model_final.keras"

IMG_SIZE = (224, 224)

# IMPORTANT:
# This must match EXACTLY the folder order used during training
CLASS_NAMES = [
    "Black_Sooty_Mold",
    "Blight_Disease",
    "Leaf_Gall_Disease",
    "Yellow_leaf_spots"
]

# =========================
# 2) Load model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# 3) Prediction function
# =========================
def predict_leaf(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array, verbose=0)[0]

    class_index = np.argmax(preds)
    confidence = preds[class_index]

    return CLASS_NAMES[class_index], confidence, preds


# =========================
# 4) Run from CLI
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide image path")
        print("Example:")
        print("  python predict_cinnamon_leaf.py test_leaf.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    label, confidence, probs = predict_leaf(image_path)

    print("\n🔍 Prediction Result")
    print("-------------------")
    print(f"🟢 Disease       : {label}")
    print(f"🟢 Confidence    : {confidence * 100:.2f}%")

    print("\n📊 Class Probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        print(f"  {cls:20s}: {p * 100:.2f}%")
