# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import cv2  # OpenCV untuk image processing
import numpy as np  # NumPy untuk operasi array/matrix
import joblib  # Untuk load saved model
import os  # Untuk file path operations
import sys  # Untuk system operations
import matplotlib.pyplot as plt  # Untuk visualisasi/plotting
from matplotlib.patches import Rectangle  # Untuk menggambar rectangle overlay
from manual_features import extract_features_manual, knn_manual, preprocess_image, lbp_manual, glcm_manual  # Import fungsi manual

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# Definisi nama kelas untuk 3 kategori jalan
CLASS_NAMES = ["Jalan Tidak Rusak", "Jalan Lubang", "Jalan Retak"]

# Warna untuk setiap kelas (format BGR untuk OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green - Jalan Tidak Rusak
    1: (0, 0, 255),    # Red - Jalan Lubang
    2: (0, 165, 255),  # Orange - Jalan Retak
}


# ============================================================================
# FUNCTION:  EXTRACT FEATURES WITH INTERMEDIATE RESULTS
# ============================================================================
def extract_features_with_intermediate(image_bgr, apply_denoise=True, denoise_method='gaussian'):
    """
    Ekstrak fitur dari gambar DAN simpan hasil intermediate untuk visualisasi.
    
    Fungsi ini melakukan ekstraksi fitur secara manual dengan menyimpan
    hasil dari setiap tahap preprocessing untuk keperluan visualisasi.
    
    Args:
        image_bgr (np.ndarray): Gambar input dalam format BGR (dari cv2.imread)
        apply_denoise (bool): Aktifkan/nonaktifkan pengurangan noise
        denoise_method (str): Metode denoising ('gaussian'/'bilateral'/'nlm')
    
    Returns:
        tuple: (feature_vector, intermediate)
            - feature_vector: Array 1D berisi 6 features (contrast, correlation, energy, homogeneity, mean, std)
            - intermediate: Dictionary berisi gambar hasil tiap step preprocessing
    """
    # ========================================
    # STEP 1: GRAYSCALE & RESIZE
    # ========================================
    # Preprocess:  resize ke 64x64 dan convert ke grayscale (tanpa denoise dulu)
    grayscale = preprocess_image(image_bgr, target_size=(64, 64), apply_denoise=False, denoise_method=denoise_method)
    
    # ========================================
    # STEP 2: NOISE REDUCTION
    # ========================================
    # Apply denoising jika enabled (gunakan metode yang dipilih)
    if apply_denoise:
        if denoise_method == 'gaussian': 
            # Gaussian Blur:  fast, mild denoising
            # Kernel size (3,3) = ukuran filter blur
            # 0 = sigmaX (calculated from kernel size)
            denoised = cv2.GaussianBlur(grayscale, (3, 3), 0)
        
        elif denoise_method == 'bilateral':
            # Bilateral Filter: slower, better edge preservation
            # d=5 = diameter neighborhood
            # sigmaColor=50 = filter sigma in color space
            # sigmaSpace=50 = filter sigma in coordinate space
            denoised = cv2.bilateralFilter(grayscale, d=5, sigmaColor=50, sigmaSpace=50)
        
        elif denoise_method == 'nlm':
            # Non-local Means: best quality, slowest
            # h=10 = filter strength (higher = more denoising)
            denoised = cv2.fastNlMeansDenoising(grayscale, h=10)
        
        else:
            # Fallback:  jika method unknown, pakai grayscale apa adanya
            denoised = grayscale. copy()
    else:
        # Jika denoising disabled, langsung copy grayscale
        denoised = grayscale.copy()
    
    # ========================================
    # STEP 3: LOCAL BINARY PATTERN (LBP)
    # ========================================
    # Extract LBP dari denoised image
    # radius=1 = jarak ke neighbor
    # num_points=8 = jumlah sampling points di sekitar pixel
    lbp_image = lbp_manual(denoised, radius=1, num_points=8)
    
    # ========================================
    # STEP 4: GLCM (GRAY-LEVEL CO-OCCURRENCE MATRIX)
    # ========================================
    # Compute GLCM dari LBP image
    # distance=1 = jarak antar pixel yang dipasangkan
    # angle=0 = direction horizontal (0 derajat)
    # gray_levels=256 = jumlah gray levels (0-255)
    glcm_matrix = glcm_manual(lbp_image, distance=1, angle=0, gray_levels=256)
    
    # Normalize GLCM ke range 0-255 untuk visualisasi
    # glcm_matrix berisi probability (0-1), kalikan 255 untuk display
    glcm_vis = (glcm_matrix * 255).astype(np.uint8)
    
    # ========================================
    # STEP 5: EXTRACT HARALICK FEATURES
    # ========================================
    # Import fungsi untuk extract 4 Haralick features dari GLCM
    from manual_features import glcm_features
    
    # Extract 4 features:  contrast, correlation, energy, homogeneity
    haralick_features = glcm_features(glcm_matrix)
    
    # ========================================
    # STEP 6: FIRST-ORDER STATISTICS
    # ========================================
    # Hitung mean dan std dari denoised image
    mean_intensity = denoised.mean()  # Rata-rata intensitas pixel
    std_intensity = denoised.std()    # Standard deviation intensitas pixel
    
    # ========================================
    # STEP 7: COMBINE ALL FEATURES
    # ========================================
    # Gabungkan 4 Haralick features + 2 first-order statistics = 6 features total
    feature_vector = np. append(haralick_features, [mean_intensity, std_intensity])
    
    # ========================================
    # STEP 8: COLLECT INTERMEDIATE RESULTS
    # ========================================
    # Simpan hasil tiap step dalam dictionary untuk visualisasi nanti
    # GLCM tidak disimpan karena tidak ditampilkan di GUI
    intermediate = {
        'grayscale': grayscale,  # Hasil step 1
        'denoised': denoised,    # Hasil step 2
        'lbp': lbp_image,        # Hasil step 3
    }
    
    return feature_vector, intermediate


# ============================================================================
# FUNCTION:  PREDICT IMAGE WITH MANUAL KNN
# ============================================================================
def predict_image_manual(image_path, model_path, apply_denoise=True, denoise_method='gaussian', verbose=True):
    """
    Prediksi kelas dari single image menggunakan trained manual KNN model.
    
    Fungsi ini melakukan prediksi kondisi jalan (tidak rusak/lubang/retak)
    dari gambar menggunakan model KNN yang telah dilatih sebelumnya.
    
    Args:
        image_path (str): Path ke file gambar
        model_path (str): Path ke saved model (.joblib)
        apply_denoise (bool): Aktifkan/nonaktifkan pengurangan noise
        denoise_method (str): Metode denoising yang digunakan
        verbose (bool): Tampilkan informasi progress
    
    Returns:
        tuple: (prediction, class_name, confidence, distances, intermediate)
            - prediction: Predicted class ID (0/1/2)
            - class_name: Predicted class name (string)
            - confidence: Confidence score (0-1)
            - distances: Distances ke k nearest neighbors
            - intermediate: Dictionary dengan intermediate images
    """
    # ========================================
    # LOAD MODEL
    # ========================================
    # Cek apakah file model exists
    if not os.path. exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model dari file . joblib
    model_data = joblib.load(model_path)
    
    # Extract training data dan parameters dari model
    X_train = model_data['X_train']      # Training features (scaled)
    y_train = model_data['y_train']      # Training labels
    k = model_data['k']                  # Number of neighbors
    scaler = model_data['scaler']        # StandardScaler untuk normalisasi
    
    # Print info model jika verbose enabled
    if verbose:
        print(f"✅ Model loaded from: {model_path}")
        print(f"   Training samples:  {X_train.shape[0]}")
        print(f"   K value: {k}")
        print()
    
    # ========================================
    # READ IMAGE
    # ========================================
    # Cek apakah file image exists
    if not os. path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image menggunakan OpenCV (format BGR)
    img_bgr = cv2.imread(image_path)
    
    # Cek apakah image berhasil di-read
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Print info image jika verbose enabled
    if verbose:
        print(f"✅ Image loaded:  {image_path}")
        print(f"   Image shape: {img_bgr.shape}")  # (height, width, channels)
        print()
    
    # ========================================
    # EXTRACT FEATURES
    # ========================================
    # Print preprocessing info
    if verbose:
        print(f"🔧 Extracting features...")
        print(f"   Noise reduction: {'Enabled' if apply_denoise else 'Disabled'}")
        if apply_denoise:
            print(f"   Denoise method: {denoise_method}")
    
    # Extract features DAN intermediate results untuk visualisasi
    features, intermediate = extract_features_with_intermediate(
        img_bgr, 
        apply_denoise=apply_denoise,
        denoise_method=denoise_method
    )
    
    # Scale features menggunakan scaler dari training
    # reshape(1, -1) = convert 1D array jadi 2D (1 row, n columns)
    # [0] = ambil row pertama (balik jadi 1D)
    features_scaled = scaler.transform(features.reshape(1, -1))[0]
    
    # Print feature info
    if verbose:
        print(f"✅ Features extracted: {len(features)} dimensions")
        print()
    
    # ========================================
    # PREDICT
    # ========================================
    if verbose:
        print(f"🔮 Making prediction...")
    
    # Predict menggunakan manual KNN
    # Returns: predicted label dan distances ke k neighbors
    prediction, distances = knn_manual(X_train, y_train, features_scaled, k=k)
    
    # Convert prediction ID ke class name
    class_name = CLASS_NAMES[prediction]
    
    # ========================================
    # CALCULATE CONFIDENCE
    # ========================================
    # Confidence = 1 / (1 + average_distance)
    # Semakin dekat neighbors, semakin tinggi confidence
    avg_distance = np.mean(distances)
    confidence = 1.0 / (1.0 + avg_distance)
    
    # Print completion
    if verbose:
        print(f"✅ Prediction completed!")
        print()
    
    return prediction, class_name, confidence, distances, intermediate


# ============================================================================
# FUNCTION: VISUALIZE WITH PREPROCESSING STEPS (WITH LBP HISTOGRAM)
# ============================================================================
def visualize_prediction_with_preprocessing(image_path, prediction, class_name, confidence, distances, 
                                           intermediate, show_plot=True, save_path=None):
    """
    Visualize hasil prediksi DENGAN preprocessing steps (5 panels).
    
    Layout: 2 rows x 3 columns
    Row 1: Original | Grayscale | Denoised
    Row 2: LBP Histogram | Prediction Result | (empty)
    
    Args: 
        image_path: Path ke image
        prediction: Predicted class ID
        class_name: Predicted class name
        confidence: Confidence score
        distances: Distances to neighbors
        intermediate: Dict dengan intermediate images
        show_plot: Show matplotlib window
        save_path: Path untuk save hasil (optional)
    """
    # ========================================
    # READ ORIGINAL IMAGE
    # ========================================
    # Read image dalam BGR (OpenCV default)
    img_bgr = cv2.imread(image_path)
    # Convert BGR ke RGB untuk matplotlib display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ========================================
    # CREATE FIGURE
    # ========================================
    # Create figure dengan 2 rows, 3 columns
    # figsize=(18, 12) = lebar 18 inch, tinggi 12 inch
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Set overall title di atas semua subplots
    # y=0.98 = posisi vertikal (0=bottom, 1=top)
    fig.suptitle(f'Pothole Detection Pipeline - {os.path.basename(image_path)}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # ========================================
    # ROW 1, COL 1: ORIGINAL IMAGE
    # ========================================
    # axes[0, 0] = row 0, column 0
    axes[0, 0].imshow(img_rgb)  # Display RGB image
    axes[0, 0].set_title('① Original Image', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')  # Hide axis ticks
    
    # ========================================
    # ROW 1, COL 2: GRAYSCALE
    # ========================================
    # Display grayscale dengan colormap gray
    axes[0, 1].imshow(intermediate['grayscale'], cmap='gray')
    axes[0, 1].set_title('② Grayscale (64x64)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    
    # ========================================
    # ROW 1, COL 3: DENOISED
    # ========================================
    # Display denoised image (setelah Gaussian Blur)
    axes[0, 2].imshow(intermediate['denoised'], cmap='gray')
    axes[0, 2].set_title('③ Denoised (Gaussian Blur)', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    
    # ========================================
    # ROW 2, COL 1: LBP HISTOGRAM (BAR CHART)
    # ========================================
    # Hitung histogram dari LBP image
    # Untuk uniform LBP dengan 8 points = 10 bins
    lbp_image = intermediate['lbp']
    n_bins = 10  # 8 uniform patterns + 1 non-uniform + 1 extra
    
    # Hitung histogram yang dinormalisasi
    hist, _ = np.histogram(
        lbp_image.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    # Buat bar chart
    bars = axes[1, 0].bar(range(n_bins), hist, color='steelblue', edgecolor='black', alpha=0.8, width=0.8)
    axes[1, 0].set_xlabel('LBP Pattern Bin', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frekuensi', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('④ LBP Histogram', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(range(n_bins))
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Tambahkan nilai di atas setiap bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # ========================================
    # ROW 2, COL 2: PREDICTION RESULT
    # ========================================
    # ========================================
    # ROW 2, COL 2: PREDICTION RESULT
    # ========================================
    # Display original image sebagai background
    axes[1, 1].imshow(img_rgb)
    
    # ---- ADD TEXT OVERLAY ----
    # Get image dimensions
    height, width = img_rgb.shape[:2]
    
    # Calculate overlay height (35% dari tinggi image)
    rect_height = int(height * 0.35)
    
    # Create semi-transparent black rectangle di atas image
    # (0, 0) = top-left corner
    # width = lebar rectangle
    # rect_height = tinggi rectangle
    # facecolor='black' = warna hitam
    # alpha=0.75 = 75% opaque (25% transparent)
    rect = Rectangle((0, 0), width, rect_height, 
                     facecolor='black', alpha=0.75, edgecolor='none')
    axes[1, 1].add_patch(rect)
    
    # ---- PREDICTION TEXT ----
    # Tentukan warna text berdasarkan prediction
    color_map = {0: 'lime', 1: 'red', 2: 'orange'}
    text_color = color_map.get(prediction, 'white')
    
    # Text: class name
    # (width/2, rect_height*0.20) = posisi X, Y
    # ha='center' = horizontal alignment center
    # va='center' = vertical alignment center
    axes[1, 1].text(width/2, rect_height*0.20, f"🏷️ {class_name}", 
                   ha='center', va='center', 
                   fontsize=16, fontweight='bold', color=text_color)
    
    # ---- CONFIDENCE TEXT ----
    # Tentukan warna berdasarkan confidence level
    # >70% = lime (hijau), 50-70% = yellow (kuning), <50% = red (merah)
    confidence_color = 'lime' if confidence > 0.7 else 'yellow' if confidence > 0.5 else 'red'
    
    # Text: confidence percentage
    # {confidence:.1%} = format as percentage with 1 decimal (e.g., 85.2%)
    axes[1, 1].text(width/2, rect_height*0.45, f"💯 {confidence:.1%}", 
                   ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=confidence_color)
    
    # ---- DISTANCE TEXT ----
    # Text: average distance to neighbors
    # {np.mean(distances):.4f} = format dengan 4 decimal places
    axes[1, 1].text(width/2, rect_height*0.65, 
                   f"📏 Avg Dist: {np.mean(distances):.4f}", 
                   ha='center', va='center', 
                   fontsize=11, color='white')
    
    # ---- FEATURE COUNT TEXT ----
    # Text: jumlah features yang di-extract
    # style='italic' = text miring
    axes[1, 1].text(width/2, rect_height*0.85, 
                   f"📊 6 features extracted", 
                   ha='center', va='center', 
                   fontsize=10, color='lightgray', style='italic')
    
    # Set title untuk panel prediction result
    axes[1, 1].set_title('⑤ Prediction Result', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # ========================================
    # ROW 2, COL 3: HIDE (NOT USED)
    # ========================================
    # Hide panel terakhir karena tidak digunakan
    axes[1, 2].axis('off')
    
    # ========================================
    # FINALIZE LAYOUT
    # ========================================
    # Adjust spacing between subplots automatically
    plt.tight_layout()
    
    # ========================================
    # SAVE IF REQUESTED
    # ========================================
    if save_path:
        # Save figure ke file
        # dpi=150 = dots per inch (resolution)
        # bbox_inches='tight' = crop whitespace around figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to:  {save_path}")
    
    # ========================================
    # SHOW OR CLOSE
    # ========================================
    if show_plot:
        # Show matplotlib window (blocking - tunggu user close window)
        plt.show()
    else:
        # Close figure without showing (untuk batch processing)
        plt.close()


# ============================================================================
# FUNCTION: VISUALIZE SIMPLE (2 PANELS)
# ============================================================================
def visualize_prediction_simple(image_path, prediction, class_name, confidence, distances, 
                                show_plot=True, save_path=None):
    """
    Visualize simple (2 panels): Original + Result
    Untuk quick preview tanpa preprocessing steps
    """
    # Read image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create figure dengan 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========================================
    # PANEL 1: ORIGINAL IMAGE
    # ========================================
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # ========================================
    # PANEL 2: PREDICTION RESULT
    # ========================================
    axes[1].imshow(img_rgb)
    
    # Add text overlay (sama seperti panel 6 di visualisasi lengkap)
    height, width = img_rgb.shape[:2]
    rect_height = int(height * 0.25)
    rect = Rectangle((0, 0), width, rect_height, 
                     facecolor='black', alpha=0.7, edgecolor='none')
    axes[1].add_patch(rect)
    
    # Prediction text
    color_map = {0: 'lime', 1: 'red', 2: 'orange'}
    text_color = color_map.get(prediction, 'white')
    
    axes[1].text(width/2, rect_height*0.3, f"🏷️ {class_name}", 
                ha='center', va='center', 
                fontsize=20, fontweight='bold', color=text_color)
    
    # Confidence text
    confidence_color = 'lime' if confidence > 0.7 else 'yellow' if confidence > 0.5 else 'red'
    axes[1].text(width/2, rect_height*0.6, f"💯 Confidence: {confidence:.2%}", 
                ha='center', va='center', 
                fontsize=16, color=confidence_color)
    
    # Distance info
    axes[1].text(width/2, rect_height*0.85, 
                f"📏 Avg Distance: {np.mean(distances):.4f}", 
                ha='center', va='center', 
                fontsize=12, color='white')
    
    axes[1].set_title('Prediction Result', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overall title
    fig.suptitle(f'Pothole Detection Result - {os.path.basename(image_path)}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt. savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()


# ============================================================================
# FUNCTION:  PRINT PREDICTION RESULT (TERMINAL OUTPUT)
# ============================================================================
def print_prediction_result(image_path, prediction, class_name, confidence, distances):
    """
    Print hasil prediksi ke terminal dengan format yang rapi
    """
    # Print separator
    print("=" * 70)
    print("🎯 PREDICTION RESULT")
    print("=" * 70)
    print()
    
    # Image info
    print(f"📸 Image: {os.path. basename(image_path)}")
    print(f"   Full path: {image_path}")
    print()
    
    # Prediction info
    print(f"🏷️  Prediction: {class_name}")
    print(f"   Class ID: {prediction}")
    print()
    
    # Confidence score
    print(f"💯 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print()
    
    # Distances to nearest neighbors
    print(f"📏 Distances to nearest neighbors:")
    for i, dist in enumerate(distances, 1):
        print(f"   Neighbor {i}: {dist:.4f}")
    print()
    
    # Distance statistics
    print(f"   Average distance: {np.mean(distances):.4f}")
    print(f"   Min distance: {np.min(distances):.4f}")
    print(f"   Max distance: {np.max(distances):.4f}")
    print()
    print("=" * 70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """
    Fungsi utama untuk menjalankan prediksi dengan command-line interface.
    
    Fungsi ini menyediakan antarmuka command-line untuk melakukan prediksi
    kondisi jalan dari gambar dengan berbagai opsi kustomisasi seperti
    metode denoising, format visualisasi, dan penyimpanan hasil.
    """
    import argparse
    
    # ========================================
    # PARSE COMMAND LINE ARGUMENTS
    # ========================================
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict pothole detection using manual KNN')
    
    # Positional argument: image path
    # nargs='?' = optional (0 or 1 argument)
    parser.add_argument('image_path', nargs='?', help='Path to image file')
    
    # Optional arguments
    parser.add_argument('--model', default=None, help='Path to model file')
    parser.add_argument('--no-denoise', action='store_true', help='Disable noise reduction')
    parser.add_argument('--denoise-method', default='gaussian', choices=['gaussian', 'bilateral', 'nlm'],
                        help='Denoising method')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show visualization window')
    parser.add_argument('--save', default=None, help='Save visualization to file')
    parser.add_argument('--simple', action='store_true', help='Use simple 2-panel visualization (no preprocessing)')
    
    # Parse arguments
    args = parser. parse_args()
    
    # ========================================
    # SETUP PATHS
    # ========================================
    # Get script directory dan project root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script is
    PROJECT_ROOT = os.path.dirname(BASE_DIR)  # One level up (project root)
    
    # Determine model path
    if args.model:
        # Jika user provide custom model path
        model_path = args.model
    else:
        # Default model path
        model_path = os.path.join(PROJECT_ROOT, "models", "manual_knn_model.joblib")
    
    # ========================================
    # CHECK MODEL EXISTS
    # ========================================
    if not os.path.exists(model_path):
        print(f"❌ Model not found:  {model_path}")
        print(f"   Please train the model first using train_manual. py")
        return
    
    # ========================================
    # DETERMINE IMAGE PATH
    # ========================================
    if args.image_path is None:
        # Jika tidak ada argument, pakai default test image
        test_image_path = os.path.join(PROJECT_ROOT, "data", "validation", "jalan_lubang", "445.jpg")
        
        # Check if default image exists
        if not os. path.exists(test_image_path):
            print(f"❌ Default test image not found: {test_image_path}")
            print(f"   Please provide image path as argument:")
            print(f"   python predict_manual.py <image_path>")
            return
    else:
        # Gunakan path yang di-provide user
        test_image_path = args.image_path
    
    # ========================================
    # CHECK IMAGE EXISTS
    # ========================================
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found: {test_image_path}")
        return
    
    # ========================================
    # SETUP DENOISE FLAG
    # ========================================
    # args.no_denoise = True berarti user mau disable denoising
    # apply_denoise = NOT no_denoise (inverse logic)
    apply_denoise = not args.no_denoise
    
    # ========================================
    # PRINT HEADER
    # ========================================
    print("\n" + "=" * 70)
    print("🚀 MANUAL KNN PREDICTION WITH VISUALIZATION")
    print("=" * 70)
    print()
    
    # ========================================
    # RUN PREDICTION
    # ========================================
    try:
        # Predict dengan intermediate results
        prediction, class_name, confidence, distances, intermediate = predict_image_manual(
            test_image_path, 
            model_path, 
            apply_denoise=apply_denoise,
            denoise_method=args.denoise_method,
            verbose=True  # Print progress
        )
        
        # Print hasil ke terminal
        print_prediction_result(test_image_path, prediction, class_name, confidence, distances)
        
        # ========================================
        # VISUALIZE
        # ========================================
        if args.simple:
            # Simple 2-panel view (no preprocessing)
            visualize_prediction_simple(
                test_image_path, prediction, class_name, confidence, distances,
                show_plot=not args.no_show,  # NOT no_show = show if flag not set
                save_path=args.save
            )
        else:
            # Full pipeline view (6 panels dengan preprocessing)
            visualize_prediction_with_preprocessing(
                test_image_path, prediction, class_name, confidence, distances, intermediate,
                show_plot=not args.no_show,
                save_path=args.save
            )
    
    except Exception as e: 
        # Jika ada error, print error message dan traceback
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ENTRY POINT
# ============================================================================
# __name__ == "__main__" = True jika script dijalankan directly
# False jika di-import sebagai module
if __name__ == "__main__":
    main()