# Vision_Based_Weld_Defect_Inspection

An automated conveyor system that uses real-time computer vision and deep learning to inspect weld defects on workpieces. 

## Features
* **Automated Conveyor Control:** Real-time synchronization between moving workpieces and image acquisition.
* **Hybrid ML Pipeline:** Integrates custom feature extraction with deep learning backbone architectures for rapid inference.
* **Surface Defect Classification:** Identifies common welding flaws.

## Tech Stack & Hardware

### Software & ML
* **Language:** Python
* **ML Frameworks:** TensorFlow / Keras
* **Architectures:** ResNet, XGBoost
* **Computer Vision:** OpenCV
* **Data Handling & Analysis:** NumPy, Pandas, Matplotlib

## 🧠 Machine Learning Architecture & Pipeline

The system employs a **hybrid Transfer Learning + XGBoost Architecture**. Instead of relying solely on end-to-end deep neural networks, the system leverages a fine-tuned ResNet50V2 backbone for feature extraction and an optimized XGBoost decision tree for final binary classification.

### Pipeline Breakdown

1. **Preprocessing & Data Augmentation:**
   * Images are standardized to **300 × 300 × 3** RGB tensors.
   * Dynamic spatial augmentations are applied via `tf.keras.Sequential` (random rotations, zoom, and contrast adjustments).
2. **Deep Feature Extraction (ResNet50V2):**
   * Pre-trained ResNet50V2 backbone initialized with ImageNet weights.
   * Concatenated `GlobalAveragePooling2D` and `GlobalMaxPooling2D` layers paired with `BatchNormalization` and `Dropout (0.5)` for strong regularization.
   * Fine-tuned top layers using a two-stage Adam optimizer schedule (lr=0.01, then lr=0.00001 for top 20 layers).
   * Outputs a 256-dimensional feature vector per image.
3. **Classification via XGBoost:**
   * Features are normalized using `StandardScaler`.
   * An **XGBClassifier** (n_estimators=100, max_depth=4, reg_lambda=10) evaluates the feature vectors to make the final classification decision.




