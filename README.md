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

## Working


 ### Step-by-Step System Flow

#### 1. System Initialization
* **Serial Port Configuration:** The Python controller initializes a serial communication channel with the Arduino board (`COM3` at 9600 baud).
* **Model Pipeline Load:** Loads the saved hybrid weights:
  1. **Keras ResNet Backbone:** Extracted bottleneck layer outputs 256-D feature representations.
  2. **StandardScaler:** Pre-fitted scaler for standardizing feature vectors.
  3. **XGBoost Classifier:** Pre-trained model for final prediction (`good weld` vs. `defects`).
* **Camera Warm-Up:** Initializes the OpenCV video capture feed (`cv2.VideoCapture`).

#### 2. Hardware Triggering (Arduino --> Python)
1. Workpieces move continuously down the conveyor belt.
2. An optical/proximity sensor wired to the Arduino detects a workpiece positioned beneath the camera station.
3. The Arduino transmits a `"TRIGGER"` string across the serial link.
4. Python listens asynchronously on the serial line.

#### 3. Real-Time Vision & Hybrid ML Inference
Once the trigger signal is received:
1. **Frame Capture:** OpenCV reads the current frame from the camera stream.
2. **Preprocessing:** Resizes the image to 300*300*3$ and normalizes pixel values.
3. **Deep Feature Extraction:** Passes the image array through the ResNet feature layer to extract a 256-D vector.
4. **Feature Scaling & Prediction:** Applies `scaler.transform()` and passes the scaled vector to `xgb_model.predict()`.

#### 4. Automated Sorting (Python --> Arduino)
1. **Decision Encoding:** The prediction index maps to a single-character command:
   * `'D'` --> **Defect Detected**
   * `'G'` --> **Good Weld**
2. **Serial Transmission:** Python encodes and transmits the single-byte message back to the Arduino.
3. **Actuator Action:**
   * **If `'D'`:** Arduino powers an sorting actuator (e.g., pneumatic pusher or servo gate) to divert the defective part off the belt.
   * **If `'G'`:** Arduino allows the workpiece to pass through unaffected.

 




