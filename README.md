# ✋ Hand Gesture Recognition System

## 📌 Overview

This project was developed as part of the **Prodigy InfoTech Machine Learning Internship (Task 04)**.

The objective of this project is to develop a **Hand Gesture Recognition System** capable of accurately identifying and classifying different hand gestures from image data using Deep Learning techniques.

The model is trained on the **LeapGestRecog Dataset** and deployed using **Streamlit** for an interactive user experience.

---

## 🚀 Features

- Hand Gesture Recognition using CNN
- Image Upload and Prediction
- Confidence Score Display
- Deep Learning Based Classification
- Data Augmentation Support
- Streamlit Web Application
- Responsive and Modern UI
- Real-Time Prediction System

---

## 📂 Dataset

**Dataset Name:** LeapGestRecog

**Source:**  
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

### Gesture Classes

| Class ID | Gesture |
|-----------|----------|
| 1 | Palm ✋ |
| 2 | L 🤟 |
| 3 | Fist ✊ |
| 4 | Fist Moved 👊 |
| 5 | Thumb 👍 |
| 6 | Index ☝️ |
| 7 | OK 👌 |
| 8 | Palm Moved 🖐️ |
| 9 | C 🤏 |
| 10 | Down 👇 |

---

## 🧠 Model Architecture

The Convolutional Neural Network (CNN) consists of:

- Conv2D Layers
- MaxPooling Layers
- Dropout Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

### Data Augmentation Techniques

- Rotation
- Zoom
- Width Shift
- Height Shift
- Horizontal Flip
- Shear Transformation

These techniques improve the model's ability to generalize on unseen images.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Streamlit
- Pillow

---

## 📊 Model Performance

| Metric | Result |
|----------|----------|
| Training Accuracy | XX.XX% |
| Validation Accuracy | XX.XX% |
| Test Accuracy | XX.XX% |

> Replace the values above with your final results.

---

## 📸 Project Screenshots

### Home Page

Add screenshot here

### Prediction Result

Add screenshot here

### Accuracy Graph

Add screenshot here

### Loss Graph

Add screenshot here

---

## 📁 Project Structure

```text
Prodigy_ML_Task04
│
├── dataset/
├── models/
│   └── hand_gesture_model.keras
│
├── screenshots/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── training_history.csv
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/ghanshyampatel13455-hub/Prodigy_ML_Task04.git
```

### Navigate to Project Folder

```bash
cd Prodigy_ML_Task04
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python -m streamlit run app.py
```

---

## 🎯 Internship Task

**Organization:** Prodigy InfoTech

**Domain:** Machine Learning

**Task 04:**

Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

---

## 👨‍💻 Author

### GHANSHYAM PATEL

**B.Tech Artificial Intelligence & Data Science**

**Machine Learning Intern @ Prodigy InfoTech**

### Connect With Me

🔗 LinkedIn:  
https://www.linkedin.com/in/ghanshyam-patel-3a8679307

💻 GitHub:  
https://github.com/ghanshyampatel13455-hub

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
