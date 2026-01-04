
def preprocess_input(X, mean, std):
    """
    Preprocess input images for model inference.
    
    Args:
        X: Input images, shape (N, 32, 32)
        mean: Training set mean
        std: Training set std
    
    Returns:
        X_processed: Normalized images, shape (N, 32, 32, 1)
    """
    # Standardization
    X_norm = (X - mean) / std
    
    # Add channel dimension
    X_norm = X_norm[..., np.newaxis]
    
    return X_norm

# Usage in predict_fn:
# X_processed = preprocess_input(X_input, mean=0.814117, std=0.738695)
# predictions = model.predict(X_processed)
