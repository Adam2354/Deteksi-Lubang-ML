import os
import glob
import numpy as np
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

from extract_features import FeatureExtractorLBP

CLASS_MAP = {
    "jalan_tidak_rusak": 0,
    "jalan_lubang": 1,
    "jalan_retak": 2,
}

def load_dataset(base_dir: str, extractor: FeatureExtractorLBP) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Peringatan: folder tidak ditemukan: {class_dir}")
            continue

        # Baca semua jpg/png, silakan tambahin ekstensi lain kalau perlu
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []
        for p in patterns:
            image_paths.extend(glob.glob(os.path.join(class_dir, p)))

        print(f"Loading {len(image_paths)} images from {class_dir} (label={label})")

        for img_path in image_paths:
            try:
                features = extractor.extract_from_path(img_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr

def train_and_evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values = (1, 3, 5, 7)
):
    # Scaling fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_k = None
    best_acc = -1.0
    best_model = None

    for k in k_values:
        print(f"\nTraining KNN with k={k}")
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="uniform"
        )  # [web:69]

        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        print(f"Accuracy (k={k}): {acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["jalan_tidak_rusak", "jalan_lubang", "jalan_retak"]
        ))

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn

    print(f"\nBest k: {best_k} with accuracy={best_acc:.4f}")
    return best_model, scaler, best_k, best_acc

def main():
    # Inisialisasi extractor (samakan config dengan waktu infer)
    extractor = FeatureExtractorLBP(
        resize_width=256,
        resize_height=256,
        lbp_radius=1,
        lbp_points=8,
        lbp_method="uniform"  # Uniform LBP: (lbp_points + 2) dimensional feature vector
    )

    # Load train & test
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    train_dir = os.path.join(PROJECT_ROOT, "data", "train")
    test_dir = os.path.join(PROJECT_ROOT, "data", "test")
    
    print("Loading training data...")
    X_train, y_train = load_dataset(train_dir, extractor)
    print("Loading test data...")
    X_test, y_test = load_dataset(test_dir, extractor)

    print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

    # Train & evaluate KNN
    model, scaler, best_k, best_acc = train_and_evaluate_knn(
        X_train, y_train, X_test, y_test,
        k_values=(1, 3, 5, 7, 9, 11, 13)
    )

    # Save model + scaler
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"knn_lbp_hist_k{best_k}.joblib")
    scaler_path = os.path.join("models", f"scaler_lbp_hist_k{best_k}.joblib")

    dump(model, model_path)
    dump(scaler, scaler_path)

    print(f"Saved best KNN model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()