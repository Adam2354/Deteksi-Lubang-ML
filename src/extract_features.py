"""
Modul untuk ekstraksi fitur hybrid menggunakan LBP Histogram dan GLCM.

Modul ini mengimplementasikan ekstraksi fitur gabungan (hybrid) yang terdiri dari:
1. LBP Histogram (10 features) - menangkap distribusi pola tekstur lokal
2. GLCM features (4 features) - menangkap hubungan spasial antar pixel
3. Statistical features (3 features) - mean, std, entropy
Total: 17 features untuk klasifikasi kondisi jalan
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class FeatureExtractorHybrid:
    def __init__(
        self,
        resize_width: int = 256,
        resize_height: int = 256,
        lbp_radius: int = 1,
        lbp_points: int = 8,
        lbp_method: str = "uniform",
        glcm_distances = (1,),
        glcm_angles = (0,),
        glcm_levels: int = 256
    ):
        """
        Inisialisasi hybrid feature extractor.
        
        Args:
            resize_width (int): Lebar target untuk resize gambar
            resize_height (int): Tinggi target untuk resize gambar
            lbp_radius (int): Radius untuk LBP (jarak ke neighbor pixels)
            lbp_points (int): Jumlah sampling points di sekitar center pixel
            lbp_method (str): Metode LBP - "uniform" untuk rotation-invariant histogram
            glcm_distances (tuple): Jarak antar pixel untuk GLCM
            glcm_angles (tuple): Sudut untuk GLCM (dalam radian)
            glcm_levels (int): Jumlah gray levels untuk GLCM
        """
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_method = lbp_method
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.glcm_levels = glcm_levels

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess gambar: resize dan konversi ke grayscale.
        
        Langkah preprocessing:
        1. Resize gambar ke ukuran standar untuk konsistensi
        2. Konversi dari BGR (OpenCV default) ke grayscale
        
        Args:
            img_bgr (np.ndarray): Gambar input dalam format BGR
            
        Returns:
            np.ndarray: Gambar grayscale hasil preprocessing
        """
        # Resize gambar ke ukuran standar
        img_resized = cv2.resize(
            img_bgr,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA
        )
        # Konversi BGR ke grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return gray

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """
        Menghitung Local Binary Pattern dari gambar grayscale.
        
        LBP menangkap pola tekstur lokal dengan cara:
        1. Membandingkan intensitas center pixel dengan neighbors
        2. Membuat binary pattern dari perbandingan tersebut
        3. Menghasilkan nilai LBP untuk setiap pixel
        
        Args:
            gray (np.ndarray): Gambar grayscale input
            
        Returns:
            np.ndarray: LBP image (uint8)
        """
        # Hitung LBP menggunakan scikit-image
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_points,
            R=self.lbp_radius,
            method=self.lbp_method
        )
        # Normalisasi LBP ke range [0, glcm_levels-1] dan konversi ke uint8
        lbp_norm = cv2.normalize(
            lbp, None, alpha=0, beta=self.glcm_levels - 1,
            norm_type=cv2.NORM_MINMAX
        )
        lbp_uint8 = lbp_norm.astype(np.uint8)
        return lbp_uint8

    def _compute_lbp_histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Menghitung histogram dari LBP image sebagai feature vector.
        
        Histogram menangkap distribusi pola tekstur:
        - Bin rendah: pola tidak teratur (lubang/retak)
        - Bin tinggi: pola seragam (jalan mulus)
        
        Untuk LBP uniform dengan 8 points, menghasilkan 10 bins:
        - Bin 0-8: uniform patterns (pola teratur)
        - Bin 9: non-uniform patterns (pola tidak teratur)
        
        Args:
            lbp_image (np.ndarray): LBP image hasil dari _compute_lbp()
            
        Returns:
            np.ndarray: Histogram LBP (10 features untuk uniform, atau 2^P untuk default)
        """
        # Tentukan jumlah bins berdasarkan metode LBP
        if self.lbp_method == "uniform":
            n_bins = self.lbp_points + 2  # 10 bins untuk 8-point uniform LBP
        else:
            n_bins = 2 ** self.lbp_points
        
        # Hitung histogram yang dinormalisasi (density=True)
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        return hist.astype(np.float32)

    def _compute_glcm_features(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Menghitung GLCM features dari grayscale image.
        
        GLCM (Gray-Level Co-occurrence Matrix) menangkap hubungan spasial antar pixel:
        - Contrast: perbedaan intensitas antara pixel dan neighbor-nya
        - Correlation: korelasi linear antara pixel pairs
        - Energy: uniformitas atau homogenitas tekstur
        - Homogeneity: kedekatan distribusi GLCM ke diagonal
        
        Args:
            img_gray (np.ndarray): Gambar grayscale (uint8)
            
        Returns:
            np.ndarray: 4 GLCM features (contrast, correlation, energy, homogeneity)
        """
        # Hitung GLCM matrix
        glcm = graycomatrix(
            img_gray,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=self.glcm_levels,
            symmetric=True,
            normed=True
        )
        
        # Ekstrak 4 properti GLCM
        feature_list = []
        for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
            props = graycoprops(glcm, prop_name)
            feature_list.extend(props.ravel())
        
        return np.array(feature_list, dtype=np.float32)

    def extract_from_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Ekstraksi hybrid features dari gambar BGR.
        
        Menggabungkan 3 jenis fitur:
        1. LBP Histogram (10 features) - distribusi pola tekstur lokal
        2. GLCM features (4 features) - hubungan spasial antar pixel
        3. Statistical features (3 features) - mean, std, entropy
        
        Total: 17 features untuk klasifikasi
        
        Args:
            img_bgr (np.ndarray): Gambar input dalam format BGR (dari cv2.imread atau video frame)
            
        Returns:
            np.ndarray: Feature vector 1D dengan shape (17,)
        """
        # Step 1: Preprocess - resize dan konversi ke grayscale
        gray = self._preprocess(img_bgr)
        
        # Step 2: Hitung LBP image
        # LBP menangkap pola tekstur lokal dengan membandingkan center pixel dengan neighbors
        lbp_image = self._compute_lbp(gray)
        
        # Step 3: Ekstrak LBP Histogram (10 features)
        # Histogram menangkap distribusi pola tekstur untuk diskriminasi jalan rusak vs mulus
        lbp_hist = self._compute_lbp_histogram(lbp_image)
        
        # Step 4: Ekstrak GLCM features dari grayscale (4 features)
        # GLCM menangkap hubungan spasial untuk deteksi tekstur beraturan/tidak beraturan
        glcm_features = self._compute_glcm_features(gray)
        
        # Step 5: Ekstrak statistical features (3 features)
        # Mean, std, dan entropy untuk menangkap karakteristik global gambar
        img_flat = gray.ravel().astype(np.float32)
        mean_val = np.mean(img_flat)
        std_val = np.std(img_flat)
        
        # Hitung entropy dari histogram
        hist_stat, _ = np.histogram(img_flat, bins=32, range=(0, 255), density=True)
        hist_stat = hist_stat + 1e-12  # Tambah epsilon untuk hindari log(0)
        entropy_val = -np.sum(hist_stat * np.log2(hist_stat))
        stat_features = np.array([mean_val, std_val, entropy_val], dtype=np.float32)
        
        # Step 6: Gabungkan semua features
        # Total: 10 (LBP) + 4 (GLCM) + 3 (stats) = 17 features
        all_features = np.concatenate([lbp_hist, glcm_features, stat_features])
        return all_features

    def extract_from_path(self, img_path: str) -> np.ndarray:
        """
        Ekstraksi features dari file gambar.
        
        Args:
            img_path (str): Path ke file gambar
            
        Returns:
            np.ndarray: Feature vector 1D dengan shape (17,)
            
        Raises:
            ValueError: Jika gagal membaca gambar
        """
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Gagal baca gambar: {img_path}")
        return self.extract_from_array(img_bgr)