Markdown# 👁️ Eye State Detector (Open vs. Closed Eyes)

An end-to-end Deep Learning & Computer Vision application designed to detect open and closed eyes in real-time across static images, video clips, and webcam feeds. Built using TensorFlow/Keras, OpenCV, and an interactive Streamlit web interface.

---

## 🚀 Features
* **🖼️ Image Analysis:** Automatically detects faces, aligns and crops eye regions, and classifies them as Open or Closed with confidence scores.
* **🎬 Video Processing:** Upload video clips to analyze eye states frame-by-frame, compute blink rates, detect long closures, and visualize probabilities over time using interactive charts.
* **📷 Real-time Camera Feed:** Take snapshots via webcam to instantly inspect eye states.
* **🛡️ Smart Fallback Detector:** Uses YuNet (OpenCV DNN) for accurate facial landmark detection, with automatic fallback to Haar cascades if needed.

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV, MediaPipe / YuNet
* **Web UI & Visualization:** Streamlit, Altair, Pandas
* **Language:** Python

---

## 📁 Project Structure
```text
📦 deep_learning_open-and-close-eyes
│
├── app.py                      # Main Streamlit application
├── cnn_eyes.ipynb              # Jupyter Notebook for model training & experimentation
├── requirements.txt            # Python dependencies
├── eye_model.keras             # Trained CNN model weights (or best_model.h5)
└── face_detection_yunet_2023mar.onnx # YuNet face detection model
⚙️ Installation & Running LocallyClone the repository:Bashgit clone [https://github.com/ahmedgable/open-and-close-eyes.git](https://github.com/ahmedgable/open-and-close-eyes.git)
cd open-and-close-eyes
Install dependencies:Bashpip install -r requirements.txt
Run the Streamlit app:Bashstreamlit run app.py
📊 Model ArchitectureThe custom Convolutional Neural Network (CNN) consists of:
3x Convolutional & Max-Pooling layers for spatial feature extraction.Dropout layer (0.2) for regularization.
Dense layers with Sigmoid activation for binary classification (Open vs. Closed).
Trained on grayscale images resized to $224 \times 224$ pixels.

👤 AuthorMahmoudGitHub: @ahmedgable
---

### إزاي تضيفه للمشروع عندك؟
1. افتح فولدر المشروع على جهازك.
2. انشئ ملف جديد وسمّه `README.md`.
3. انسخ الكود اللي فوق وحطه جوه الملف واحفظه.
4. ارفع الملف لجيت هاب بالأوامر دي:
   ```bash
   git add README.md
   git commit -m "Add professional README.md"
   git push origin main
