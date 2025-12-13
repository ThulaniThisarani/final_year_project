import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

img_size = 300  # EfficientNetB3 uses bigger image size
batch_size = 16
epochs = 40
dataset_dir = r"C:\Users\Thulani\Desktop\project\dataset"

# -------------------------------------------------------------------
# DATA AUGMENTATION (carefully tuned)
# -------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation"
)

# -------------------------------------------------------------------
# LOAD EfficientNetB3 (MUCH stronger than EfficientNetB0)
# -------------------------------------------------------------------
base = EfficientNetB3(weights="imagenet", include_top=False,
                      input_shape=(img_size, img_size, 3))

base.trainable = False  # first freeze for stabilizing training

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------------------
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# -------------------------------------------------------------------
# FIRST TRAINING PHASE (stabilize)
# -------------------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# -------------------------------------------------------------------
# SECOND TRAINING PHASE: UNFREEZE LAST 50 LAYERS FOR FINE-TUNING
# -------------------------------------------------------------------
for layer in base.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),  # VERY IMPORTANT FOR FINE TUNING
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=callbacks
)

model.save("cinnamon_best_accuracy.h5")
print("Training Completed Successfully!")