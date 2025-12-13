# ============================================================
# CINNAMON LEAF DISEASE CLASSIFICATION – FINAL ROBUST TRAINING
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE_1 = 20
EPOCHS_STAGE_2 = 30

DATASET_DIR = r"C:\Users\Thulani\Desktop\project\dataset"  # change if needed

# -----------------------------
# DATA GENERATOR (NO VALIDATION)
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

NUM_CLASSES = train_gen.num_classes
print("Detected classes:", train_gen.class_indices)

# -----------------------------
# MODEL: EfficientNetB0 (CNN)
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# -----------------------------
# CUSTOM CLASSIFIER HEAD
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ============================================================
# STAGE 1 – TRAIN CLASSIFIER ONLY
# ============================================================
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStage 1 Training (Classifier Head Only)\n")
model.fit(
    train_gen,
    epochs=EPOCHS_STAGE_1
)

# ============================================================
# STAGE 2 – FINE-TUNE CNN (CRITICAL STEP)
# ============================================================
base_model.trainable = True

# Freeze first layers, fine-tune deeper ones
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStage 2 Training (Fine-Tuning CNN)\n")
model.fit(
    train_gen,
    epochs=EPOCHS_STAGE_2
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save("cinnamon_leaf_disease_final.keras")
print("\n✅ Training complete. Model saved as cinnamon_leaf_disease_final.keras")