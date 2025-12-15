import os
import sys
import cv2
import numpy as np
from joblib import load

from extract_features import FeatureExtractorLBP

# Sesuaikan dengan CLASS_MAP di train_knn.py
LABEL_MAP = {
    0: "jalan_tidak_rusak",
    1: "jalan_lubang",
    2: "jalan_retak",
}

def load_model_and_scaler(models_dir: str):
    # Sesuaikan nama file dengan yang tadi kesimpan (k=3)
    model_path = os.path.join(models_dir, "knn_lbp_hist_k3.joblib")
    scaler_path = os.path.join(models_dir, "scaler_lbp_hist_k3.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_path}")

    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_image.py path_ke_gambar")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Gambar tidak ditemukan: {img_path}")
        sys.exit(1)

    # Setup extractor (harus sama config dengan training)
    extractor = FeatureExtractorLBP(
        resize_width=256,
        resize_height=256,
        lbp_radius=1,
        lbp_points=8,
        lbp_method="uniform"  # Uniform LBP: (lbp_points + 2) dimensional feature vector
    )

    # Load model + scaler
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")

    model, scaler = load_model_and_scaler(models_dir)

    # Ekstrak fitur
    features = extractor.extract_from_path(img_path)          # shape (n_features,)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Prediksi
    pred_label_int = int(model.predict(features_scaled)[0])
    pred_label_str = LABEL_MAP.get(pred_label_int, f"unknown({pred_label_int})")

    # (Opsional) ambil juga jarak tetangga terdekat
    # distances, indices = model.kneighbors(features_scaled, n_neighbors=3)

    print(f"Gambar : {img_path}")
    print(f"Prediksi kelas : {pred_label_str} (label={pred_label_int})")

    # (Opsional) tampilkan gambar dengan teks prediksi
    img_bgr = cv2.imread(img_path)
    if img_bgr is not None:
        text = pred_label_str
        cv2.putText(img_bgr, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Prediksi", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
