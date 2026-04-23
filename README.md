# 🧠 MNIST Handwritten Digit Classification using Deep Learning

## 👨‍💻 Group Members
- **FC211012** – M.H.M Azeem  
- **FC211025** – W.M.M.C.B Wijesundara  
- **FC211030** – M.M Siyas  
- **FC211028** – M.R.M.R Fasri  

---

## 📌 Project Overview
This project focuses on building a **Convolutional Neural Network (CNN)** model to classify handwritten digits (0–9) using the **MNIST dataset**.

A **Streamlit web application** is developed to allow users to upload an image of a handwritten digit and receive real-time predictions with confidence scores.

---

## 🎯 Objectives
- Develop a CNN model for digit classification  
- Achieve high accuracy on the MNIST dataset  
- Deploy the model using a web-based interface  
- Enable users to upload images and get predictions  

---

## 🧱 Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Streamlit  
- Matplotlib  

---

## 📂 Project Structure
```
Handwritten_digit_Classification_using_deep_learning/
│
├── models/
│   └── mnist_tensorflow_prediction_model.keras
│
├── src/
│   ├── app.py              # Streamlit frontend + backend
│   ├── model_utils.py     # Preprocessing & prediction logic
│   └── train_model.py     # Model training script
│
├── README.md
└── requirements.txt
```

---

## 🧠 Model Architecture (LeNet-5 Inspired)
- Conv2D (6 filters, 5×5, activation = tanh)  
- Average Pooling  
- Conv2D (16 filters, 5×5, activation = tanh)  
- Average Pooling  
- Flatten  
- Dense (120 neurons, activation = tanh)  
- Dense (84 neurons, activation = tanh)  
- Output Layer (10 neurons, activation = softmax)  

---

## 📊 Dataset
- **MNIST Handwritten Digits Dataset**  
- 60,000 training images  
- 10,000 testing images  
- Image size: **28 × 28 grayscale**

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd Handwritten_digit_Classification_using_deep_learning
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

_or manually:_
```bash
pip install tensorflow==2.13.0 streamlit numpy pillow opencv-python matplotlib
```

---

## 🚀 Run the Application
```bash
cd src
streamlit run app.py
```

Then open your browser:
```
http://localhost:8501
```

---

## 📸 How It Works
1. Upload a handwritten digit image  
2. Image is preprocessed:
   - Converted to grayscale  
   - Resized to 28×28  
   - Normalized  
3. The trained CNN model predicts the digit  
4. Prediction and confidence scores are displayed  

---