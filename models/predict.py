# For threshold logic

def predict_with_threshold(y_prob, threshold=0.5):
    return (y_prob >= threshold).astype(int)