import os
import cv2
import numpy as np
from joblib import load

from extract_features import FeatureExtractorLBPGLCM

LABEL_MAP = {
    0: "jalan_tidak_rusak",
    1: "jalan_lubang",
    2: "jalan_retak",
}

def load_model_and_scaler(models_dir: str):
    model_path = os.path.join(models_dir, "knn_lbp_glcm_k3.joblib")
    scaler_path = os.path.join(models_dir, "scaler_lbp_glcm_k3.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_path}")

    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler

def main():
    # init extractor
    extractor = FeatureExtractorLBPGLCM(
        resize_width=256,
        resize_height=256,
        lbp_radius=1,
        lbp_points=8,
        lbp_method="default",
        glcm_distances=(1,),
        glcm_angles=(0,),
        glcm_levels=256,
    )

    # paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, "models")

    model, scaler = load_model_and_scaler(models_dir)

    # 0 = webcam default, kalau mau file video: ganti jadi path file [web:110][web:114]
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Gagal buka kamera/video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Definisikan ROI area jalan (misal bawah-tengah frame)
        # Silakan adjust sesuai posisi kamera di kendaraan [web:108][web:109]
        roi_top = int(h * 0.5)
        roi_bottom = h
        roi_left = int(w * 0.25)
        roi_right = int(w * 0.75)

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # Ekstrak fitur dari ROI
        features = extractor.extract_from_array(roi)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Prediksi kelas
        pred_label_int = int(model.predict(features_scaled)[0])
        pred_label_str = LABEL_MAP.get(pred_label_int, f"unknown({pred_label_int})")

        # Gambar kotak ROI + label di frame [web:113][web:117][web:120][web:123]
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv2.putText(frame, pred_label_str, (roi_left + 10, roi_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Deteksi Kondisi Jalan (KNN + LBP+GLCM)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
