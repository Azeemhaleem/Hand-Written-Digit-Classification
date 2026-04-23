import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

def load_model(model_path='../models/mnist_tensorflow_prediction_model.keras'):
    """
    Load your trained MNIST model
    """
    try:
        # Load your model
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"✅ Model summary: {model.summary()}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def preprocess_image(image):
    """
    Preprocess image to match MNIST format (28x28 grayscale)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # If RGBA, use alpha channel or convert to grayscale
        if image.shape[2] == 4:
            # Check if we should use alpha channel
            if np.mean(image[:, :, 3]) < 128:
                gray = 255 - image[:, :, 3]
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors if background is dark (MNIST digits are white on black)
    if np.mean(resized) > 127:
        resized = 255 - resized
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model (1, 28, 28, 1)
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    # For display, scale back to 0-255
    display_img = (normalized * 255).astype('uint8')
    
    return reshaped, display_img

def predict_digit(model, image):
    """
    Make prediction on a single image
    Returns: predicted digit, confidence, all predictions
    """
    # Preprocess the image
    processed_img, _ = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get predicted digit and confidence
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get all predictions sorted by confidence
    all_predictions = []
    for i, prob in enumerate(predictions[0]):
        all_predictions.append({
            'digit': int(i),
            'confidence': float(prob)
        })
    
    # Sort by confidence (descending)
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predicted_digit, confidence, all_predictions

# Test function to verify model loading
def test_model():
    """Test if model loads correctly"""
    try:
        model = load_model()
        print(f"✅ Model loaded successfully!")
        print(f"✅ Input shape: {model.input_shape}")
        print(f"✅ Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Run a quick test
    test_model()