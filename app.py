import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Optional: HEIC support (if you decide to allow HEIC uploads)
try:
    import pillow_heif
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224

# IMPORTANT:
# These class names must be in the SAME ORDER as during training.
# Typically, flow_from_directory() orders them alphabetically by folder name.
CLASS_NAMES = [
    "Black_Sooty_Mold",
    "Blight_Disease",
    "Leaf_Gall_Disease",
    "Yellow_leaf_spots"
]

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_cinnamon_model():
    # Try main filename; adjust if your model has a different name
    try:
        model = load_model("cinnamon_leaf_disease_classifier.h5")
    except Exception:
        model = load_model("best_cinnamon_model.h5")
    return model

model = load_cinnamon_model()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def load_image(uploaded_file):
    """
    Takes a Streamlit UploadedFile and returns a PIL.Image object.
    Supports JPG/PNG and (optionally) HEIC if pillow-heif is installed.
    """
    filename = uploaded_file.name.lower()

    # Basic formats (JPEG, PNG, etc.)
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file).convert("RGB")
        return image

    # HEIC support (if installed)
    if filename.endswith(".heic"):
        if not HEIC_AVAILABLE:
            raise ValueError(
                "HEIC image uploaded but pillow-heif is not installed. "
                "Install with `pip install pillow-heif` or upload JPG/PNG instead."
            )
        data = uploaded_file.read()
        heif_file = pillow_heif.read_heif(io.BytesIO(data))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        ).convert("RGB")
        return image

    # Fallback: try PIL directly
    image = Image.open(uploaded_file).convert("RGB")
    return image


def preprocess_image(pil_image):
    """
    Resize and scale image to match the model input.
    """
    img = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array


def predict_disease(pil_image):
    """
    Run prediction on a PIL image and return predicted class + probabilities.
    """
    img_batch = preprocess_image(pil_image)
    preds = model.predict(img_batch)[0]  # shape (4,)
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = preds[predicted_index] * 100
    return predicted_class, confidence, preds


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Cinnamon Leaf Disease Detector", layout="centered")

st.title("🌿 Cinnamon Leaf Disease Detection")
st.write(
    "Upload a cinnamon leaf image to detect whether it has:\n"
    "- **Yellow Leaf Spots**\n"
    "- **Black Sooty Mold**\n"
    "- **Leaf Gall Disease**\n"
    "- **Blight Disease**"
)

uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG/PNG, HEIC optional)",
    type=["jpg", "jpeg", "png", "heic"]
)

if uploaded_file is not None:
    try:
        # Load and display image
        pil_img = load_image(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        # Run prediction
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, probabilities = predict_disease(pil_img)

        st.subheader("Prediction")
        st.markdown(
            f"**Disease:** `{predicted_class}`  \n"
            f"**Confidence:** `{confidence:.2f}%`"
        )

        # Show probability for each class
        st.subheader("Class Probabilities")
        prob_dict = {cls: float(f"{p*100:.2f}") for cls, p in zip(CLASS_NAMES, probabilities)}
        st.write(prob_dict)

        # Optional: nice bar chart
        st.bar_chart(
            data=probabilities,
            x=None,
            y=None,
        )

        st.caption(
            "Note: This is a deep learning model. Always confirm with an expert agronomist or plant pathologist."
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Try converting HEIC to JPG/PNG and re-uploading if the issue persists.")
else:
    st.info("👆 Upload a leaf image to get started.")
