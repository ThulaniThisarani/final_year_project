<div align="center">

# 🌿 Cinnamon Leaf Disease Classifier

*Transfer Learning · EfficientNetB0 · 4 Disease Classes · Progressive Fine-Tuning*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF8C00?style=for-the-badge&logo=tensorflow&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-Recommended-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-5DCAA5?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [🌱 Overview](#-overview)
- [🔬 Disease Classes](#-disease-classes)
- [📁 Project Structure](#-project-structure)
- [📦 Requirements](#-requirements)
- [🗂️ Dataset Setup](#️-dataset-setup)
- [🚀 Usage](#-usage)
- [🏗️ Model Architecture](#️-model-architecture)
- [⚙️ Training Configuration](#️-training-configuration)
- [💾 Output Files](#-output-files)

---

## 🌱 Overview

A fine-tuned *EfficientNetB0* model that classifies cinnamon leaf diseases into 4 categories. Trained with strong data augmentation, label smoothing, and a two-stage progressive fine-tuning strategy. Designed for high accuracy on small agricultural datasets.

---

## 🔬 Disease Classes

| &nbsp; | Class | Description |
|:---:|:---|:---|
| 🖤 | Black_Sooty_Mold | Fungal growth caused by honeydew-secreting insects |
| 🟠 | Blight_Disease | Rapid browning and death of leaf tissue |
| 🌸 | Leaf_Gall_Disease | Abnormal growths or swellings on leaf surface |
| 🟡 | Yellow_leaf_spots | Yellowing patches indicating infection or deficiency |

---

## 📁 Project Structure


cinnamon-leaf-disease/
│
├── 📂 dataset/                            ← training images (not in repo)
│   ├── Black_Sooty_Mold/
│   ├── Blight_Disease/
│   ├── Leaf_Gall_Disease/
│   └── Yellow_leaf_spots/
│
├── 🐍 train9.py                           ← model training script
├── 🐍 app.py                              ← single image prediction
├── 🐍 convert.py                          ← HEIC → JPG converter
│
├── ✅ best_cinnamon_model.keras           ← generated (best checkpoint)
├── ✅ cinnamon_disease_model_final.keras  ← generated (final model)
│
└── 📄 README.md


---

## 📦 Requirements

bash
pip install tensorflow scikit-learn pillow pillow-heif numpy


| Package | Purpose |
|:---|:---|
| tensorflow | Model training & inference |
| scikit-learn | Evaluation metrics |
| pillow | Image loading |
| pillow-heif | HEIC image support (iPhone photos) |
| numpy | Array operations |

> *Note:* GPU is strongly recommended. CPU training will be very slow for Stage 2 fine-tuning.

---

## 🗂️ Dataset Setup

Organize your images in the following structure before training:


dataset/
├── Black_Sooty_Mold/
│   ├── img1.jpg
│   └── ...
├── Blight_Disease/
│   └── ...
├── Leaf_Gall_Disease/
│   └── ...
└── Yellow_leaf_spots/
    └── ...


### 📱 Converting iPhone HEIC Images

If your photos are from iPhone (HEIC format), convert them to JPG first:

bash
python convert.py


This recursively scans dataset/ and converts all .heic files to .jpg in-place.

---

## 🚀 Usage

### Step 1 — Train the Model

bash
python train9.py


Training runs automatically in two stages. Checkpoints and the final model are saved to disk.

### Step 2 — Predict a Single Image

bash
python app.py path/to/leaf_image.jpg


*Example output:*


✅ Model loaded successfully

🔍 Prediction Result
-------------------
🟢 Disease       : Blight_Disease
🟢 Confidence    : 94.37%

📊 Class Probabilities:
  Black_Sooty_Mold    :  1.23%
  Blight_Disease      : 94.37%
  Leaf_Gall_Disease   :  2.85%
  Yellow_leaf_spots   :  1.55%


---

## 🏗️ Model Architecture


Input (224 × 224 × 3)
        │
        ▼
Data Augmentation
(flip · rotate · zoom · contrast · brightness)
        │
        ▼
EfficientNet Preprocessing
        │
        ▼
┌───────────────────────────────┐
│   EfficientNetB0 Backbone     │  ← ImageNet pretrained
│   (frozen in Stage 1)         │
└───────────────────────────────┘
        │
        ▼
GlobalAveragePooling2D
        │
BatchNormalization
        │
Dense(512, ReLU) → Dropout(0.50)
        │
Dense(256, ReLU) → Dropout(0.40)
        │
        ▼
Dense(4, Softmax)  ← output


---

## ⚙️ Training Configuration

### Hyperparameters

| Setting | Value |
|:---|:---:|
| Image size | 224 × 224 |
| Batch size | 32 |
| Validation split | 30% |
| Label smoothing | 0.10 |
| Dropout (head) | 0.50, 0.40 |

### Two-Stage Training Strategy

#### 🧊 Stage 1 — Head Training

| | |
|:---|:---|
| Backbone | Frozen |
| Learning rate | 1e-3 |
| Max epochs | 15 |

#### 🔥 Stage 2 — Progressive Fine-Tuning

| | |
|:---|:---|
| Unfrozen layers | Last 120 of EfficientNetB0 |
| Learning rate | 1e-5 |
| Max epochs | 40 |

### Callbacks

| Callback | Monitor | Detail |
|:---|:---|:---|
| ModelCheckpoint | val_accuracy | Saves best model only |
| EarlyStopping | val_loss | Patience = 8 |
| ReduceLROnPlateau | val_loss | Factor = 0.3, Patience = 3 |

---

## 💾 Output Files

| File | Description |
|:---|:---|
| ⭐ best_cinnamon_model.keras | Best checkpoint — highest val_accuracy during training |
| 🏁 cinnamon_disease_model_final.keras | Final model after Stage 2 completes |

> 💡 *Tip:* Use best_cinnamon_model.keras for inference — it captures peak validation performance and is often more robust than the final epoch model.

---

## ⚠️ Troubleshooting

*Model predicts only one class?*
Check for class imbalance, incorrect folder mapping, or repeated images in your dataset.

*CLASS_NAMES order matters!*
The list in app.py must match the *alphabetical* folder order that TensorFlow's image_dataset_from_directory assigns automatically.

---

<div align="center">

Made with 🌿 for cinnamon leaf disease detection

</div>
