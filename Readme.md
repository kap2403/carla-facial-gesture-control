Sure! Here's a polished and clear version of your README, tailored to present well during an interview. It improves clarity, grammar, formatting, and consistency while keeping your original content and technical details intact:

---

# 🚗 CARLA Gesture Control Client

Control a vehicle in the [CARLA Simulator](https://carla.org/) using **facial gestures** detected via webcam. This project integrates **MediaPipe** for facial landmark detection and a **ResNet50-based deep neural network** for gesture classification, enabling intuitive, hands-free vehicle control.

---

## 🧠 Features

- **Facial Gesture-Based Vehicle Control**  
  Map facial expressions and head movements to vehicle actions:
  - **Tilt head left** → Turn vehicle left  
  - **Tilt head right** → Turn vehicle right  
  - **Mouth open** → Accelerate  
  - **Frown** → Brake  

- **Gear Control**  
  Control the gear (forward or reverse) with vertical head movement:
  - **Head up** → Reverse gear  
  - **Head down** → Forward gear  

---

## 📸 Dataset Creation

Capture facial gestures using your webcam to build a dataset for training.

### How to Use:

1. Set the gesture class (e.g., `right_tilt`, `left_tilt`, `mouth_open`, `frown`)  
2. Define the time duration (in seconds) for which to record data  

In `data_collection.py`, update the main block:

```python
if __name__ == "__main__":
    dataset_folder_path = "dataset/images_landmarks_dataset"
    os.makedirs(dataset_folder_path, exist_ok=True)
    class_name = "right_tilt"  # Replace with the desired gesture class
    time_limit = 50            # Time in seconds
    dataset_creation(class_name, time_limit, dataset_folder_path)
```

### Run the script:

```bash
cd src/dataset
python data_collection.py
```

---

## 🧱 Model Architecture

The gesture classification model is a hybrid deep learning architecture that combines:

- **Image features** extracted from webcam frames using a **pretrained ResNet50**
- **Landmark features** extracted via **MediaPipe**, passed through a **Multi-Layer Perceptron (MLP)**
- **Fusion of both modalities** for robust gesture recognition

### 🔍 Architecture Overview

<img src="readme_images/architecture.png" alt="Gesture Recognition Model Architecture" width="400"/>

### 🔧 Model Components

- **ResNet50 (pretrained)**  
  - Used for visual feature extraction  
  - Final FC layer removed, outputs a 2048-dimensional vector  

- **MLP for Landmark Embedding**  
  - Input: 1434-dimensional facial landmark vector  
  - Output: 64-dimensional embedded feature vector  

- **Classifier**  
  - Input: Combined 2112-dimensional vector  
  - Outputs: 5 gesture class predictions  

---

## 🏋️‍♂️ Model Training

Train the gesture classification model using the collected dataset.

### How to Use:

Update the `train.py` script with the desired model save path:

```python
if __name__ == "__main__":
    main(model_save_path="models/updated_weights")
```

### Run the training:

```bash
cd src/model
python train.py
```

---

## 🚀 Real-Time Inference & Control

Run the application to control the CARLA vehicle in real-time using facial gestures.

### Run the main control script:

```bash
python main.py
```

Make sure to load the correct model weights before starting.

---

## 📌 Notes

- Ensure your webcam is functional and accessible before running scripts.
- For best results, record data in a well-lit environment with minimal background distractions.
- The gesture classifier is based on **ResNet50** and fine-tuned for facial landmark-based inputs.

---
