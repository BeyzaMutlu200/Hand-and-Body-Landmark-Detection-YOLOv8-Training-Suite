# Hand & Holistic Landmark Detection + YOLOv8 Pipeline

This repository contains scripts to **collect hand and body landmark datasets**, **train a YOLOv8 model**, and **perform real-time detection**.
It is designed for applications like **gesture recognition, liveness detection, and hand pose estimation**.

**Note:** The scripts may appear similar in structure, but each serves a different purpose and allows you to create and test datasets in multiple ways. You can experiment with them to generate datasets that best fit your needs.

---

## 📂 Repository Structure

```
.
├─ hand.py             # Collect hand landmarks and save images + YOLO-format txt
├─ ds_landmark.py      # Collect holistic landmarks (face + body)
├─ hand_dataset.py     # Preprocess hand images and landmarks for dataset creation
├─ detect.py           # Real-time detection script for trained YOLOv8 model
├─ split_data.py       # Split dataset into train/val/test and create data.yaml
├─ yolo.py             # Train YOLOv8 model
├─ test.py             # Real-time detection using trained YOLOv8 model
├─ dataset/
│  ├─ all/             # Original images + labels
│  ├─ SplitData/       # YOLO-ready train/val/test folders
│  │  ├─ train/
│  │  │  ├─ images/
│  │  │  └─ labels/
│  │  ├─ val/
│  │  │  ├─ images/
│  │  │  └─ labels/
│  │  └─ test/
│  │     ├─ images/
│  │     └─ labels/
└─ yaml/
   └─ data.yaml         # YOLO dataset config
```

---

## 1️⃣ hand.py – Hand Landmark Dataset Collection

**Purpose:**
Capture hand images and landmarks from webcam, save images and YOLO-format annotation files.

**How it works:**

* Uses **MediaPipe Hands** to detect hand landmarks.
* Normalizes landmarks to `[0,1]` for YOLO-style annotations.
* Saves each frame as:

  * `.jpg` image
  * `.txt` annotation file in format: `class_id x y w h`

**Notes / Improvements:**

* Currently `w` and `h` are fixed (0.01), real bounding boxes can be computed from landmarks.
* Optionally filter out blurry frames using Laplacian variance.
* Ignore frames where the hand is mostly out-of-frame.

---

## 2️⃣ hand\_dataset.py – Hand Dataset Preprocessing

**Purpose:**
Prepare collected hand images and annotations for YOLO training.

**How it works:**

* Reads raw hand images and `.txt` landmark files.
* Performs normalization, cropping, and resizing as required.
* Ensures proper alignment of `.jpg` and `.txt` files.
* Saves processed dataset in `SplitData` format ready for YOLO.

---

## 3️⃣ ds\_landmark.py – Holistic Landmark Collection

**Purpose:**
Capture face + full-body landmarks for advanced gesture/liveness detection datasets.

**How it works:**

* Uses **MediaPipe Holistic** to detect:

  * Face landmarks
  * Pose landmarks
* Normalizes all landmarks to `[0,1]` coordinates.
* Saves images and `.txt` files for each frame.

**Notes:**

* Can be combined with hand landmarks for richer datasets.
* Ensure camera captures full body for consistent pose landmarks.

---

## 4️⃣ split\_data.py – Dataset Split & YOLO Preparation

**Purpose:**
Split dataset into **train, validation, and test** sets, and prepare folder structure and YOLO `data.yaml`.

**How it works:**

* Reads `.jpg` and `.txt` files from the dataset folder.
* Shuffles and splits files based on `SPLIT_DATA` ratios (default: 70% train, 20% val, 10% test).
* Creates folders:

  ```
  SplitData/train/images
  SplitData/train/labels
  SplitData/val/images
  SplitData/val/labels
  SplitData/test/images
  SplitData/test/labels
  ```
* Generates `data.yaml` for YOLOv8.

---

## 5️⃣ yolo.py – YOLOv8 Training

**Purpose:**
Train YOLOv8 model on the prepared dataset.

**How it works:**

* Loads a YOLOv8 model (e.g., `yolov8n.pt`).
* Uses `data.yaml` to know where train/val/test images and labels are.
* Trains for a set number of epochs (`epochs=10` in example).

**Output:**

* Trained weights: `runs/detect/train/weights/best.pt`
* Training logs and metrics.

---

## 6️⃣ detect.py – Real-Time Detection

**Purpose:**
Run real-time detection with the trained YOLOv8 model.

**How it works:**

* Loads the trained weights.
* Captures frames from webcam.
* Performs detection and draws bounding boxes, class labels, and confidence scores.
* Press `q` to exit.

---

## 7️⃣ test.py – Alternative Real-Time Test

**Purpose:**
Test the trained YOLOv8 model in real-time with webcam feed.

**How it works:**

* Loads `best.pt` trained weights.
* Captures webcam frames.
* Performs detection and draws bounding boxes, class labels, confidence scores.
* Press `q` to exit.

---

## 🔧 Installation & Requirements

```bash
pip install opencv-python mediapipe ultralytics numpy
```

* Python 3.8+ recommended.
* GPU recommended for YOLO training but not required for dataset collection.

---

## 🖥 Usage (Terminal / Copyable)

```bash
# 1️⃣ Collect hand landmarks
python hand.py

# 2️⃣ Preprocess hand dataset
python hand_dataset.py

# 3️⃣ Collect holistic landmarks (optional)
python ds_landmark.py

# 4️⃣ Split dataset into train/val/test and prepare YOLO format
python split_data.py

# 5️⃣ Train YOLOv8 model
python yolo.py

# 6️⃣ Run real-time detection (alternative)
python detect.py

# 7️⃣ Test YOLOv8 model in real-time using webcam
python test.py
```

---

## 🎯 Purpose

* Build **high-quality hand and holistic landmark datasets**.
* Train **YOLOv8 models** for hand gesture recognition and liveness detection.
* Test and validate models **in real-time**.
* **Note:** These scripts are similar in structure, but each is intended for experimentation. You can modify, combine, and run them to create datasets tailored to your needs and test your model performance in different scenarios.
