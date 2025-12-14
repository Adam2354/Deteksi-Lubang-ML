import os
import sys
import cv2
import numpy as np
from joblib import load

# Import manual features
from manual_features import extract_features_manual, knn_manual

LABEL_MAP = {
    0: "jalan_tidak_rusak",
    1: "jalan_lubang", 
    2: "jalan_retak",
}

def load_training_data(models_dir: str):
    """Load training features + labels (harus disave dari training manual)"""
    # Ganti scaler sama manual training data
    train_path = os.path.join(models_dir, "manual_train_features.npy")
    labels_path = os.path.join(models_dir, "manual_train_labels.npy")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data tidak ditemukan: {train_path}")
    
    X_train = np.load(train_path)
    y_train = np.load(labels_path)
    return X_train, y_train

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_manual.py path_ke_gambar")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Gambar tidak ditemukan: {img_path}")
        sys.exit(1)

    # Load manual training data (bukan model sklearn)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")
    
    X_train, y_train = load_training_data(models_dir)
    print(f"Loaded {len(X_train)} training samples")

    # MANUAL FEATURE EXTRACTION (no library!)
    img_bgr = cv2.imread(img_path)
    features = extract_features_manual(img_bgr)
    print(f"Extracted features: {features.shape} = {features}")

    # MANUAL KNN PREDICTION (no sklearn!)
    pred_label, distances = knn_manual(X_train, y_train, features, k=3)
    pred_label_str = LABEL_MAP.get(pred_label, f"unknown({pred_label})")
    
    print(f"Gambar     : {img_path}")
    print(f"Prediksi   : {pred_label_str} (label={pred_label})")
    print(f"NN distances: {distances}")

    # Visualisasi
    text = pred_label_str
    cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Manual Prediksi", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
