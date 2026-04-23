🧠 MNIST Handwritten Digit Classification using Deep Learning

👨‍💻 Group Members
FC211012 – M.H.M Azeem
FC211025 – W.M.M.C.B Wijesundara
FC211030 – M.M Siyas
FC211028 – M.R.M.R Fasri

📌 Project Overview
This project focuses on building a Convolutional Neural Network (CNN) model to classify handwritten digits (0–9) using the MNIST dataset.

A Streamlit web application is developed to allow users to upload an image of a handwritten digit and receive real-time predictions.

🎯 Objectives
Develop a CNN model for digit classification
Achieve high accuracy on MNIST dataset
Deploy the model using a simple web interface
Allow users to upload images and get predictions

🧱 Technologies Used
Python
TensorFlow / Keras
NumPy
OpenCV
Streamlit
Matplotlib

📂 Project Structure
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

🧠 Model Architecture (LeNet-5 Inspired)
Conv2D (6 filters, 5×5, tanh)
Average Pooling
Conv2D (16 filters, 5×5, tanh)
Average Pooling
Flatten
Dense (120 neurons, tanh)
Dense (84 neurons, tanh)
Output Layer (10 neurons, softmax)

📊 Dataset
MNIST Handwritten Digits Dataset
60,000 training images
10,000 testing images
Image size: 28 × 28 grayscale

⚙️ Installation & Setup
1️⃣ Create virtual environment
python -m venv .venv

2️⃣ Activate environment
# Windows
.venv\Scripts\activate

3️⃣ Install dependencies
pip install tensorflow==2.13.0 streamlit numpy pillow opencv-python matplotlib

🚀 Run the Application
cd src
streamlit run app.py

Open browser:
http://localhost:8501

📸 How It Works
User uploads a handwritten digit image
Image is preprocessed:
Converted to grayscale
Resized to 28×28
Normalized
Model predicts the digit
Result and confidence scores are displayed

✅ Features
User-friendly web interface
Real-time digit prediction
Confidence visualization
Image preprocessing pipeline

⚠️ Known Issues & Fixes
.h5 model compatibility issues with newer TensorFlow versions
Fixed by using .keras format for model saving

📈 Expected Results
Accuracy: ~98–99% on MNIST dataset

📝 Conclusion
This project demonstrates how deep learning models can be integrated into a web application to provide real-time predictions. The combination of CNN and Streamlit enables an efficient and user-friendly solution for handwritten digit recognition.