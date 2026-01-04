
"""
PREDICT FUNCTION FOR SUBMISSION

This function loads the trained model and makes predictions on new data.
"""

import numpy as np
from tensorflow import keras

# Preprocessing parameters (from training)
MEAN = 0.8154657483100891
STD = 0.7396610975265503

def predict_fn(X_input):
    """
    Make predictions on input data.
    
    Args:
        X_input: numpy array of shape (N, 32, 32)
                 Raw grayscale images (unnormalized)
    
    Returns:
        y_a: numpy array of shape (N,) with predicted classes [0-9]
        y_b: numpy array of shape (N,) with predicted classes [0-31]
        y_c: numpy array of shape (N,) with predicted values [0-1]
    """
    
    # 1. Preprocess input (same as training)
    X_norm = (X_input - MEAN) / STD
    
    # 2. Add channel dimension
    X_norm = X_norm[..., np.newaxis]  # (N, 32, 32) → (N, 32, 32, 1)
    
    # 3. Load model
    model = keras.models.load_model('model_group99.h5')  # Update with your group ID
    
    # 4. Make predictions
    predictions = model.predict(X_norm, verbose=0)
    pred_a, pred_b, pred_c = predictions
    
    # 5. Convert to required format
    y_a = np.argmax(pred_a, axis=1).astype(np.int32)  # Class predictions
    y_b = np.argmax(pred_b, axis=1).astype(np.int32)  # Class predictions
    y_c = pred_c.squeeze().astype(np.float32)         # Regression predictions
    
    return y_a, y_b, y_c


# Example usage:
if __name__ == "__main__":
    # Test on training data
    data = np.load('train.npz')
    X_test = data['X'][:10]  # First 10 samples
    
    y_a, y_b, y_c = predict_fn(X_test)
    
    print("Predictions:")
    print(f"  Target A shape: {y_a.shape}, dtype: {y_a.dtype}")
    print(f"  Target B shape: {y_b.shape}, dtype: {y_b.dtype}")
    print(f"  Target C shape: {y_c.shape}, dtype: {y_c.dtype}")
    print(f"\nSample predictions:")
    print(f"  A: {y_a[:5]}")
    print(f"  B: {y_b[:5]}")
    print(f"  C: {y_c[:5]}")
