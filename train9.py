"""
Robust Cinnamon Leaf Disease Classifier (4 classes)
Method: Transfer Learning (EfficientNet) + Strong Augmentation + Label Smoothing + Progressive Fine-Tuning

Dataset structure (your structure):
dataset/
  Black_Sooty_Mold/
  Blight_Disease/
  Leaf_Gall_Disease/
  Yellow_leaf_spots/

Run:
  python train_cinnamon_disease.py

Notes:
- GPU recommended (not CPU-friendly by design).
- Saves:
    best_cinnamon_model.keras   (best val_accuracy)
    cinnamon_disease_model_final.keras (final after fine-tuning)
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 0) Basic Setup (GPU + Determinism)
# =========================
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Allow GPU memory growth (prevents TF from grabbing all VRAM at once)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("⚠️ Could not set memory growth:", e)

print("✅ TensorFlow version:", tf.__version__)
print("✅ GPUs available:", gpus)


# =========================
# 1) Config
# =========================
DATASET_DIR = "dataset"     # <-- change if needed
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.30

# Training schedule
EPOCHS_STAGE1 = 15          # head training
EPOCHS_STAGE2 = 40          # fine-tuning
UNFREEZE_LAST_N_LAYERS = 120

# Optimization
INIT_LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

# Regularization tricks
LABEL_SMOOTHING = 0.1
DROPOUT_1 = 0.50
DROPOUT_2 = 0.40

# =========================
# 2) Validate dataset folder
# =========================
if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}\n"
        "Make sure your folder name is correct and contains class subfolders."
    )


# =========================
# 3) Load Dataset (Directory -> tf.data)
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"   # one-hot
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\n✅ Classes detected:", class_names)
print("✅ Number of classes:", NUM_CLASSES)

if NUM_CLASSES < 2:
    raise ValueError("Need at least 2 classes. Check your dataset folder structure.")


# =========================
# 4) Improve pipeline performance (prefetch/cache)
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# =========================
# 5) Strong Augmentation (leaf-friendly)
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.30),
    layers.RandomBrightness(0.20),
], name="augmentation")


# =========================
# 6) Build Model (EfficientNet + Robust Head)
# =========================
# EfficientNet preprocess: expects RGB in [0..255], converts internally
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base_model.trainable = False  # Stage 1: freeze backbone

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)

# Backbone
x = base_model(x, training=False)

# Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(DROPOUT_1)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(DROPOUT_2)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs, name="CinnamonDisease_EfficientNetB0")

model.summary()


# =========================
# 7) Compile Stage 1
# =========================
loss_fn = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

model.compile(
    optimizer=Adam(learning_rate=INIT_LR_STAGE1),
    loss=loss_fn,
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        "best_cinnamon_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# =========================
# 8) Train Stage 1 (Head training)
# =========================
print("\n==============================")
print("🚀 Stage 1: Training classifier head (backbone frozen)")
print("==============================\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)


# =========================
# 9) Stage 2: Progressive Fine-Tuning
# =========================
print("\n==============================")
print("🔥 Stage 2: Fine-tuning deeper layers")
print("==============================\n")

base_model.trainable = True

# Freeze all but last N layers of EfficientNet
if UNFREEZE_LAST_N_LAYERS > len(base_model.layers):
    UNFREEZE_LAST_N_LAYERS = len(base_model.layers)

for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

print(f"✅ Unfroze last {UNFREEZE_LAST_N_LAYERS} layers of EfficientNet backbone.")

# Re-compile with lower LR
model.compile(
    optimizer=Adam(learning_rate=LR_STAGE2),
    loss=loss_fn,
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)


# =========================
# 10) Save final model
# =========================
model.save("cinnamon_disease_model_final.keras")
print("\n✅ Final model saved as: cinnamon_disease_model_final.keras")
print("✅ Best checkpoint saved as: best_cinnamon_model.keras")


# =========================
# 11) Evaluation (Classification Report + Confusion Matrix)
# =========================
print("\n==============================")
print("📊 Evaluation on Validation Set")
print("==============================\n")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))


# =========================
# 12) Quick sanity checks
# =========================
unique_preds = np.unique(y_pred)
print("\n✅ Unique predicted classes:", unique_preds)
if len(unique_preds) == 1:
    print("⚠️ WARNING: Model is predicting only one class. This indicates training collapse.")
    print("   Common causes: label issues, wrong folder mapping, repeated images, or extreme class imbalance.")
else:
    print("✅ Model predicts multiple classes (good sign).")
