import os
import glob
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from manual_features import extract_features_manual, knn_manual

# Class mapping
CLASS_LABELS = {
    "jalan_tidak_rusak": 0,
    "jalan_lubang": 1,
    "jalan_retak": 2,
}

CLASS_NAMES = ["Jalan Tidak Rusak", "Jalan Lubang", "Jalan Retak"]


def load_dataset_manual(dataset_dir, apply_denoise=True, denoise_method='gaussian', verbose=True):
    """
    Load dataset dan extract features secara manual
    
    Args: 
        dataset_dir: Path ke folder dataset (train/test/validation)
        apply_denoise:  Enable noise reduction
        denoise_method:  Metode denoising ('gaussian', 'bilateral', 'nlm')
        verbose: Print progress information
    
    Returns:
        features_array: numpy array (n_samples, n_features)
        labels_array: numpy array (n_samples,)
        file_paths: list of image paths
    """
    features_list = []
    labels_list = []
    file_paths_list = []
    
    if verbose:
        print(f"📂 Loading dataset from:  {dataset_dir}")
        print(f"   Noise reduction: {'Enabled' if apply_denoise else 'Disabled'}")
        if apply_denoise:
            print(f"   Denoise method: {denoise_method}")
        print()
    
    for class_name, label in CLASS_LABELS.items():
        class_dir = os. path.join(dataset_dir, class_name)
        
        if not os.path.isdir(class_dir):
            if verbose:
                print(f"   ⚠️  Warning: Directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob. glob(os.path.join(class_dir, pattern)))
        
        if verbose:
            print(f"   Loading {len(image_paths):3d} images from '{class_name}'...")
        
        success_count = 0
        for img_path in image_paths: 
            try:
                # Read image
                img_bgr = cv2.imread(img_path)
                
                if img_bgr is None:
                    if verbose: 
                        print(f"      ⚠️  Failed to read:  {os.path.basename(img_path)}")
                    continue
                
                # Extract features
                features = extract_features_manual(
                    img_bgr, 
                    apply_denoise=apply_denoise,
                    denoise_method=denoise_method
                )
                
                features_list.append(features)
                labels_list.append(label)
                file_paths_list.append(img_path)
                success_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"      ❌ Error processing {os.path.basename(img_path)}: {e}")
        
        if verbose:
            print(f"      ✅ Successfully loaded:  {success_count}/{len(image_paths)}")
    
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)
    
    if verbose:
        print(f"\n   Total samples loaded: {len(features_array)}")
        print(f"   Feature dimensions: {features_array.shape[1]}")
        print()
    
    return features_array, labels_array, file_paths_list


def train_manual_knn(X_train, y_train, k=3):
    """
    Train manual KNN (store training data dengan normalization)
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        k: Number of neighbors
    
    Returns:
        model_data: Dictionary containing training data and parameters
    """
    # Normalize features menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Manual KNN tidak perlu explicit training
    # Simpan training data untuk prediksi nanti
    model_data = {
        'X_train': X_train_scaled,
        'y_train':  y_train,
        'k': k,
        'scaler': scaler,
        'feature_dim': X_train. shape[1],
        'num_classes': len(np.unique(y_train))
    }
    
    return model_data


def evaluate_manual_knn(model_data, X_test, y_test, verbose=True):
    """
    Evaluate manual KNN implementation dengan detailed metrics
    
    Args: 
        model_data: Dictionary dengan training data dan scaler
        X_test: Test features (n_samples, n_features)
        y_test: True labels (n_samples,)
        verbose: Print progress information
    
    Returns:
        predictions: Predicted labels
        accuracy: Accuracy score
        conf_matrix: Confusion matrix
        class_report: Classification report
    """
    # Extract data from model
    X_train = model_data['X_train']
    y_train = model_data['y_train']
    k = model_data['k']
    scaler = model_data['scaler']
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predict each test sample
    predictions = []
    
    if verbose:
        print(f"🔮 Predicting {len(X_test)} samples with k={k}...")
    
    for i, x_test in enumerate(X_test_scaled):
        if verbose and (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(X_test)} ({(i+1)/len(X_test)*100:.1f}%)")
        
        prediction, distances = knn_manual(X_train, y_train, x_test, k=k)
        predictions.append(prediction)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    class_report = classification_report(
        y_test, 
        predictions, 
        target_names=CLASS_NAMES,
        digits=4
    )
    
    if verbose:
        print(f"   ✅ Prediction completed!")
        print()
    
    return predictions, accuracy, conf_matrix, class_report


def print_detailed_results(accuracy, conf_matrix, class_report):
    """Print hasil evaluasi dengan format yang rapi"""
    print("=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print()
    
    # Accuracy
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Confusion Matrix
    print("📈 Confusion Matrix:")
    print("   (Rows = True Label, Columns = Predicted Label)")
    print()
    print("                    Predicted")
    print("                 Tidak  Lubang  Retak")
    print("              +--------+-------+-------+")
    for i, class_name in enumerate(["Tidak Rusak", "Lubang     ", "Retak      "]):
        print(f"   {class_name} |  {conf_matrix[i, 0]: 4d}  | {conf_matrix[i, 1]:4d}  | {conf_matrix[i, 2]:4d}  |")
    print("              +--------+-------+-------+")
    print()
    
    # Classification Report
    print("📋 Classification Report:")
    print(class_report)
    print("=" * 70)


def main():
    """Main training and evaluation pipeline"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING MANUAL KNN WITH IMPROVED PREPROCESSING & EVALUATION")
    print("=" * 70)
    print()
    
    # Configuration
    K_VALUE = 3
    APPLY_DENOISE = True
    DENOISE_METHOD = 'gaussian'  # Options: 'gaussian', 'bilateral', 'nlm'
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    train_dir = os.path.join(PROJECT_ROOT, "data", "train")
    test_dir = os.path.join(PROJECT_ROOT, "data", "test")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load datasets
    print("📂 STEP 1: LOADING DATASETS")
    print("-" * 70)
    X_train, y_train, train_paths = load_dataset_manual(
        train_dir, 
        apply_denoise=APPLY_DENOISE,
        denoise_method=DENOISE_METHOD,
        verbose=True
    )
    
    X_test, y_test, test_paths = load_dataset_manual(
        test_dir, 
        apply_denoise=APPLY_DENOISE,
        denoise_method=DENOISE_METHOD,
        verbose=True
    )
    
    print(f"✅ Training samples: {X_train.shape[0]} images, {X_train.shape[1]} features")
    print(f"✅ Test samples: {X_test.shape[0]} images, {X_test.shape[1]} features")
    print()
    
    # Train model
    print("🎓 STEP 2: TRAINING MODEL")
    print("-" * 70)
    print(f"Training manual KNN with k={K_VALUE}...")
    model_data = train_manual_knn(X_train, y_train, k=K_VALUE)
    print("✅ Training completed!")
    print()
    
    # Evaluate on test set
    print("📊 STEP 3: EVALUATING ON TEST SET")
    print("-" * 70)
    predictions, accuracy, conf_matrix, class_report = evaluate_manual_knn(
        model_data, X_test, y_test, verbose=True
    )
    
    # Print detailed results
    print_detailed_results(accuracy, conf_matrix, class_report)
    print()
    
    # Save model
    print("💾 STEP 4: SAVING MODEL")
    print("-" * 70)
    
    model_path = os.path.join(models_dir, "manual_knn_model.joblib")
    joblib.dump(model_data, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save features for future use
    features_path = os.path.join(models_dir, "manual_train_features.npy")
    labels_path = os.path.join(models_dir, "manual_train_labels.npy")
    
    np.save(features_path, model_data['X_train'])
    np.save(labels_path, model_data['y_train'])
    
    print(f"✅ Training features saved to: {features_path}")
    print(f"✅ Training labels saved to: {labels_path}")
    print()
    
    print("=" * 70)
    print("✅ ALL DONE!  Model training and evaluation completed successfully!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()