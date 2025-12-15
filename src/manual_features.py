import cv2
import numpy as np

def preprocess_image(image_bgr, target_size=(64, 64), apply_denoise=True, denoise_method='gaussian'):
    """
    Preprocess image dengan resize dan noise reduction
    
    Args:
        image_bgr:  Input image dalam format BGR (numpy array)
        target_size: Target size untuk resize (width, height)
        apply_denoise:  Apakah apply denoising atau tidak
        denoise_method:  Metode denoising ('gaussian', 'bilateral', 'nlm')
    
    Returns:
        Grayscale image yang sudah dipreprocess (numpy array)
    """
    # Resize image
    resized_image = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction (optional)
    if apply_denoise:
        if denoise_method == 'gaussian':
            # Gaussian Blur - Fast, mild denoising
            denoised_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
        
        elif denoise_method == 'bilateral':
            # Bilateral Filter - Slower, better edge preservation
            denoised_image = cv2.bilateralFilter(grayscale_image, d=5, sigmaColor=50, sigmaSpace=50)
        
        elif denoise_method == 'nlm':
            # Non-local Means Denoising - Best quality, slowest
            denoised_image = cv2.fastNlMeansDenoising(grayscale_image, h=10)
        
        else:
            raise ValueError(f"Unknown denoise method: {denoise_method}")
        
        return denoised_image
    
    return grayscale_image


def lbp_manual(grayscale_image, radius=1, num_points=8):
    """
    LBP (Local Binary Pattern) circular neighborhood - manual implementation
    
    Args: 
        grayscale_image:  Input grayscale image (numpy array 2D)
        radius: Radius dari circular neighborhood (default:  1)
        num_points:  Jumlah sampling points di sekitar pixel (default: 8)
    
    Returns:
        lbp_result: LBP image (numpy array 2D, dtype=uint8)
    """
    image_height, image_width = grayscale_image.shape
    lbp_result = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Generate sampling angles (circular neighborhood)
    # Sudut dari 0 hingga 2π, dibagi merata sebanyak num_points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Calculate offset coordinates untuk setiap sampling point
    # Gunakan trigonometri:  x = r*cos(θ), y = r*sin(θ)
    coordinate_offsets = [
        (int(radius * np.cos(angle)), int(radius * np.sin(angle))) 
        for angle in angles
    ]
    
    # Process setiap pixel (skip border karena tidak punya neighbor lengkap)
    for row_idx in range(radius, image_height - radius):
        for col_idx in range(radius, image_width - radius):
            
            # Pixel pusat yang akan dibandingkan dengan neighbors
            center_pixel_value = grayscale_image[row_idx, col_idx]
            binary_code = 0
            
            # Compare dengan setiap neighbor
            for bit_position, (delta_row, delta_col) in enumerate(coordinate_offsets):
                # Koordinat neighbor pixel
                neighbor_row = row_idx + delta_row
                neighbor_col = col_idx + delta_col
                
                # Border handling:  jika keluar dari image boundary
                if 0 <= neighbor_row < image_height and 0 <= neighbor_col < image_width: 
                    neighbor_pixel_value = grayscale_image[neighbor_row, neighbor_col]
                else:
                    # Gunakan nilai center sebagai fallback
                    neighbor_pixel_value = center_pixel_value
                
                # Build binary code: 
                # Jika neighbor >= center, set bit ke 1 pada posisi tertentu
                # Gunakan bitwise OR dan left shift
                binary_code |= (neighbor_pixel_value >= center_pixel_value) << bit_position
            
            # Simpan LBP code untuk pixel ini
            lbp_result[row_idx, col_idx] = binary_code
    
    return lbp_result


def glcm_manual(lbp_image, distance=1, angle=0, gray_levels=256):
    """
    GLCM (Gray-Level Co-occurrence Matrix) - manual computation
    
    Args:
        lbp_image: Input image (biasanya hasil dari LBP)
        distance: Jarak antar pixel yang dipasangkan
        angle: Sudut direction (0, 45, 90, 135, 180, 225, 270, 315)
        gray_levels: Jumlah gray level (default: 256)
    
    Returns:
        cooccurrence_matrix:  GLCM matrix (numpy array 2D, normalized)
    """
    image_height, image_width = lbp_image.shape
    cooccurrence_matrix = np. zeros((gray_levels, gray_levels), dtype=np.float32)
    
    # Tentukan direction offset berdasarkan angle
    if angle == 0:  # Horizontal (kanan)
        row_offset, col_offset = 0, distance
    elif angle == 90:  # Vertical (bawah)
        row_offset, col_offset = distance, 0
    elif angle == 45:  # Diagonal (kanan-bawah)
        row_offset, col_offset = distance, distance
    elif angle == 135:  # Diagonal (kiri-bawah)
        row_offset, col_offset = distance, -distance
    else: 
        # Untuk angle lain, hitung dengan trigonometri
        angle_rad = np.deg2rad(angle)
        row_offset = int(distance * np.sin(angle_rad))
        col_offset = int(distance * np.cos(angle_rad))
    
    # Build co-occurrence matrix
    for row_idx in range(image_height):
        for col_idx in range(image_width):
            
            # Koordinat neighbor pixel
            neighbor_row = row_idx + row_offset
            neighbor_col = col_idx + col_offset
            
            # Pastikan neighbor dalam boundary
            if 0 <= neighbor_row < image_height and 0 <= neighbor_col < image_width:
                current_pixel = lbp_image[row_idx, col_idx]
                neighbor_pixel = lbp_image[neighbor_row, neighbor_col]
                
                # Increment count untuk pasangan (current, neighbor)
                cooccurrence_matrix[current_pixel, neighbor_pixel] += 1
    
    # Normalize:  jadikan probabilitas (sum = 1)
    matrix_sum = cooccurrence_matrix.sum()
    if matrix_sum > 0:
        cooccurrence_matrix /= matrix_sum
    
    return cooccurrence_matrix


def glcm_features(cooccurrence_matrix):
    """
    Extract Haralick features dari GLCM matrix
    
    Args: 
        cooccurrence_matrix:  GLCM matrix (2D numpy array)
    
    Returns:
        features: Array berisi [contrast, correlation, energy, homogeneity]
    """
    # Create index matrices
    row_indices, col_indices = np.ogrid[
        : cooccurrence_matrix.shape[0], 
        :cooccurrence_matrix.shape[1]
    ]
    
    # Feature 1: Contrast
    # Mengukur perbedaan intensitas antar pixel
    # Nilai tinggi = gambar kasar/bergerigi (lubang jalan)
    contrast = np.sum(cooccurrence_matrix * (row_indices - col_indices) ** 2)
    
    # Feature 2: Homogeneity (Inverse Difference Moment)
    # Mengukur keseragaman tekstur
    # Nilai tinggi = tekstur seragam (jalan mulus)
    homogeneity = np.sum(cooccurrence_matrix / (1 + (row_indices - col_indices) ** 2))
    
    # Feature 3: Energy (Angular Second Moment)
    # Mengukur keseragaman distribusi GLCM
    # Nilai tinggi = pola repetitif
    energy = np.sqrt(np.sum(cooccurrence_matrix ** 2))
    
    # Feature 4: Correlation
    # Mengukur ketergantungan linear antar pixel
    mean_row = np.sum(row_indices * cooccurrence_matrix. sum(axis=1))
    mean_col = np. sum(col_indices * cooccurrence_matrix.sum(axis=0))
    
    std_row = np.sqrt(np.sum(((row_indices - mean_row) ** 2) * cooccurrence_matrix.sum(axis=1)))
    std_col = np.sqrt(np.sum(((col_indices - mean_col) ** 2) * cooccurrence_matrix.sum(axis=0)))
    
    # Prevent division by zero
    if std_row * std_col > 0:
        correlation = np.sum(
            cooccurrence_matrix * (row_indices - mean_row) * (col_indices - mean_col)
        ) / (std_row * std_col)
    else:
        correlation = 0.0
    
    return np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)


def knn_manual(training_features, training_labels, test_sample, k=3):
    """
    K-Nearest Neighbors classifier - pure NumPy implementation
    
    Args:
        training_features: Training feature matrix (n_samples, n_features)
        training_labels: Training labels array (n_samples,)
        test_sample: Single test sample (n_features,)
        k: Number of nearest neighbors
    
    Returns:
        prediction:  Predicted class label
        nearest_distances: Distances to k nearest neighbors
    """
    # Calculate Euclidean distance dari test_sample ke semua training samples
    # Distance = sqrt(sum((x_train - x_test)^2))
    squared_differences = (training_features - test_sample) ** 2
    distances = np.sqrt(squared_differences. sum(axis=1))
    
    # Find k nearest neighbors (smallest distances)
    # argpartition lebih efisien dari argsort untuk partial sorting
    nearest_indices = np.argpartition(distances, k)[:k]
    nearest_distances = distances[nearest_indices]
    
    # Get labels dari k nearest neighbors
    nearest_labels = training_labels[nearest_indices]
    
    # Majority voting:  label dengan count terbanyak
    unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)
    prediction = unique_labels[np.argmax(label_counts)]
    
    return prediction, nearest_distances


def extract_features_manual(image_bgr, apply_denoise=True, denoise_method='gaussian'):
    """
    Extract LBP + GLCM features dari gambar dengan preprocessing
    
    Pipeline:
    1. Preprocess (resize + noise reduction)
    2. LBP extraction
    3. GLCM computation dari LBP image
    4. Haralick features extraction
    5. First-order statistics (mean, std)
    
    Args:
        image_bgr: Input image dalam BGR format (numpy array)
        apply_denoise: Enable/disable noise reduction
        denoise_method: Metode denoising ('gaussian', 'bilateral', 'nlm')
    
    Returns:
        feature_vector: 1D array berisi semua features
    """
    # Step 1: Preprocess dengan noise reduction
    grayscale_image = preprocess_image(
        image_bgr, 
        target_size=(64, 64), 
        apply_denoise=apply_denoise,
        denoise_method=denoise_method
    )
    
    # Step 2: Extract LBP features
    lbp_image = lbp_manual(grayscale_image, radius=1, num_points=8)
    
    # Step 3: Compute GLCM dari LBP image
    cooccurrence_matrix = glcm_manual(lbp_image, distance=1, angle=0, gray_levels=256)
    
    # Step 4: Extract Haralick features dari GLCM
    haralick_features = glcm_features(cooccurrence_matrix)
    
    # Step 5: Extract first-order statistics
    mean_intensity = grayscale_image.mean()
    std_intensity = grayscale_image.std()
    
    # Combine all features into single vector
    feature_vector = np.append(haralick_features, [mean_intensity, std_intensity])
    
    return feature_vector