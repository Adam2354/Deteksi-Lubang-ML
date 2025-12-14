import cv2
import numpy as np
import os
from manual_features import extract_features_manual

# Path dataset lo (sesuaikan)
dataset_dir = "D:/Kuliah/Semester 7/ML/Deteksi-Lubang/data/train"
classes = ["jalan_tidak_rusak", "jalan_lubang", "jalan_retak"]
class_to_label = {cls: i for i, cls in enumerate(classes)}

X_train, y_train = [], []

for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.exists(class_dir):
        continue
        
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            features = extract_features_manual(img_bgr)
            X_train.append(features)
            y_train.append(class_to_label[class_name])
            print(f"Processed {img_name} -> class {class_name}")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training data: {X_train.shape}, labels: {y_train.shape}")

os.makedirs("models", exist_ok=True)
np.save("models/manual_train_features.npy", X_train)
np.save("models/manual_train_labels.npy", y_train)
print("Manual training data saved!")
