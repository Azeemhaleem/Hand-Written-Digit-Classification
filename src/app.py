import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 2.8rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .digit-display {
        font-size: 6rem;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🔢 MNIST Handwritten Digit Classifier</h1>', unsafe_allow_html=True)

# Sidebar for model info
with st.sidebar:
    st.markdown("### 🏗️ Model Information")
    st.markdown("""
    **Model Details:**
    - Type: CNN (Convolutional Neural Network)
    - Input: 28×28 grayscale image
    - Output: Digit 0-9
    - Location: `..models/mnist_tensorflow_prediction_model.h5`
    """)
    
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    **Expected Performance:**
    - Training Accuracy: ~99%
    - Validation Accuracy: ~98-99%
    - Test Accuracy: ~98-99%
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    1. White digit on dark background
    2. Center the digit in image
    3. Good contrast
    4. Single digit per image
    5. No rotations
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload Your Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a handwritten digit image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Upload an image containing a single digit (0-9)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Store in session state for processing
        st.session_state.uploaded_image = np.array(image)
        st.session_state.image_ready = True
    else:
        st.markdown('<div class="upload-box">📁 Drag and drop or click to upload</div>', unsafe_allow_html=True)
        st.session_state.image_ready = False
    
    # Alternative: Use sample images
    st.markdown("### 🧪 Try Sample Digits")
    sample_cols = st.columns(5)
    sample_digits = list(range(10))
    
    for idx, col in enumerate(sample_cols):
        if idx < len(sample_digits):
            with col:
                if st.button(f"{sample_digits[idx]}", use_container_width=True):
                    # Create a simple test image
                    img = np.zeros((28, 28, 3), dtype=np.uint8)
                    # In a real app, you'd have actual sample images
                    st.info(f"Selected digit {sample_digits[idx]}")
                    st.session_state.test_digit = sample_digits[idx]
                    st.session_state.use_test = True

with col2:
    st.markdown("## 🔮 Prediction Results")
    
    if 'prediction' in st.session_state and st.session_state.prediction is not None:
        # Display prediction
        st.markdown(f'''
        <div class="prediction-box">
            <div class="digit-display">{st.session_state.prediction}</div>
            <div class="confidence-text">Confidence: {st.session_state.confidence*100:.2f}%</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Show processed image
        if 'processed_image' in st.session_state:
            st.markdown("**Processed Image (28×28):**")
            st.image(st.session_state.processed_image, width=150)
        
        # Show confidence bar
        confidence_percent = st.session_state.confidence * 100
        st.progress(confidence_percent / 100, text=f"Confidence: {confidence_percent:.1f}%")
        
        # Show top predictions
        if 'all_predictions' in st.session_state:
            st.markdown("### 📈 Top Predictions")
            
            # Create a simple bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            top_n = min(5, len(st.session_state.all_predictions))
            top_preds = st.session_state.all_predictions[:top_n]
            
            digits = [str(p['digit']) for p in top_preds]
            confidences = [p['confidence'] * 100 for p in top_preds]
            
            colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(digits)))
            bars = ax.bar(digits, confidences, color=colors, edgecolor='black')
            
            # Add value labels
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{conf:.1f}%', ha='center', va='bottom')
            
            ax.set_ylabel('Confidence (%)')
            ax.set_xlabel('Digit')
            ax.set_title('Top Predictions')
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            
            # Detailed table
            st.markdown("### 📋 Detailed Results")
            for i, pred in enumerate(top_preds):
                col_a, col_b, col_c = st.columns([1, 6, 2])
                with col_a:
                    st.markdown(f"**{i+1}.**")
                with col_b:
                    st.progress(pred['confidence'])
                with col_c:
                    st.markdown(f"**{pred['digit']}** ({pred['confidence']*100:.1f}%)")
    
    else:
        st.markdown('<div class="info-box">👈 Upload an image to see predictions here</div>', unsafe_allow_html=True)
        
        # Show example
        st.markdown("### Example Output:")
        st.markdown('''
        <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 10px;">
            <div style="font-size: 4rem; font-weight: bold; color: #1E88E5;">7</div>
            <div style="color: #666; margin-top: 1rem;">Confidence: 98.5%</div>
        </div>
        ''', unsafe_allow_html=True)

# Prediction button and processing
st.markdown("---")
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        if 'image_ready' in st.session_state and st.session_state.image_ready:
            with st.spinner("🔍 Processing image..."):
                # Import model utilities
                try:
                    from model_utils import load_model, preprocess_image, predict_digit
                    
                    # Load model
                    model = load_model('../models/mnist_tensorflow_prediction_model.keras')
                    
                    # Process image
                    processed_img, display_img = preprocess_image(st.session_state.uploaded_image)
                    
                    # Make prediction
                    predicted_digit, confidence, all_predictions = predict_digit(model, st.session_state.uploaded_image)
                    
                    # Store results in session state
                    st.session_state.prediction = predicted_digit
                    st.session_state.confidence = confidence
                    st.session_state.all_predictions = all_predictions
                    st.session_state.processed_image = display_img
                    
                    # Success message
                    st.success("✅ Prediction completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.info("Make sure your model file exists at: ../models/mnist_tensorflow_prediction_model.keras")
        else:
            st.warning("⚠️ Please upload an image first!")

# Clear button
if st.button("🔄 Clear Results", type="secondary"):
    for key in ['prediction', 'confidence', 'all_predictions', 'processed_image', 'uploaded_image', 'image_ready']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>MNIST Digit Classification | Deep Learning Assignment</p>
    <p>Using your trained model: <code>mnist_tensorflow_prediction_model.h5</code></p>
</div>
""", unsafe_allow_html=True)