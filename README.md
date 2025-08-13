# 📷 Vision AI in 5 Days: From Beginner to Image Recognition Expert

## 📌 Project Overview

This project was developed as part of the **"Build an AI that Sees"** Bootcamp by **Devtown**, in collaboration with **MSIT Student Chapter** and **Google Developer Groups**.\
The goal was to **design, train, and deploy** an image recognition system using **Python, TensorFlow/Keras, and deep learning techniques** — culminating in a deployable, portfolio-ready toolkit.

The project walks through the entire computer vision workflow, from **data preprocessing** to **transfer learning optimization**, and demonstrates how to **build a real-world AI model capable of recognizing images**.

---

## 🎯 Learning Outcomes & Skills Gained

- Image preprocessing & augmentation with **OpenCV** & **TensorFlow**
- Designing **Convolutional Neural Networks (CNNs)** from scratch
- Model training, evaluation, and performance optimization
- Transfer learning with **MobileNetV2** (and comparison with custom CNN)
- Visualization of results with **Matplotlib** & **Seaborn**
- Documentation, collaboration, and GitHub portfolio readiness
- Deployable AI model for recruiters and practical applications

---

## 🗂 Dataset

- **Dataset Name**: CIFAR-10
- **Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Preprocessing Steps**:
  - Image resizing to `(32, 32)`
  - Normalization to `[0,1]` range
  - Visualization of sample images before training

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn, Scikit-learn
- **Platform:** Google Colab (GPU enabled)
- **Model Types:**
  - Custom CNN from scratch
  - Transfer Learning with MobileNetV2

---

## 🚀 Project Workflow

### **1️⃣ Data Preprocessing & Exploration**

- Downloaded dataset from TensorFlow Datasets
- Resized images to a fixed dimension
- Normalized pixel values
- Visualized a subset of dataset samples

### **2️⃣ Building & Training CNN Model**

- Created a sequential CNN model with Conv2D, MaxPooling, Flatten, Dense, Dropout layers
- Compiled with `Adam` optimizer, `categorical_crossentropy` loss
- Trained for multiple epochs with validation split

### **3️⃣ Data Augmentation & Model Evaluation**

- Applied rotation, flipping, and zoom augmentation
- Evaluated with **Accuracy, Precision, Recall, F1-score, Confusion Matrix**
- Visualized training curves & confusion matrix

### **4️⃣ Transfer Learning Optimization**

- Fine-tuned **MobileNetV2** pre-trained on ImageNet
- Compared accuracy & loss with the custom CNN
- Achieved performance boost

### **5️⃣ Documentation, Demo & Submission**

- Uploaded complete code to GitHub
- Recorded **30-sec live demo video**
- Created **5-slide evaluation presentation**
- Prepared **LinkedIn post**

---

## 📊 Model Performance

| Model       | Accuracy |
| ----------- | -------- |
| Custom CNN  | 68%      |
| MobileNetV2 | 75%      |

**Confusion Matrix for Custom CNN:**\
<img src="outputs/Custom CNN Confusion Matrix.png"></img>

---

**ROC for MobileNetV2:**\
<img src="outputs/Fine Tuned CNN ROC.png"></img>

---

## 🗂️ Repository Structure

```
📦 Vision-AI-5-Days
 ┣ 🗂 models/                                      # Saved model files (.h5)
 ┣ 🗂 outputs/                                     # Visualizations & predictions
 ┣ 📝 CIFAR_10_Vehicle_Classification.ipynb       # Jupyter Notebook
 ┣ 📝 README.md                                   # Project documentation
 ┗ 📝 requirements.txt                            # Python dependencies
```

---

## 🎥 Demo Video

<video src="outputs/Screen Record.mp4"></video>

---

## 🔗 Google Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PSewmuthu/CIFAR-10_Vehicle_Classification/blob/main/CIFAR_10_Vehicle_Classification.ipynb)

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify with attribution.
