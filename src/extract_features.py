import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class FeatureExtractorLBP:
    def __init__(
        self,
        resize_width: int = 256,
        resize_height: int = 256,
        lbp_radius: int = 1,
        lbp_points: int = 8,
        lbp_method: str = "uniform"
    ):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_method = lbp_method

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # Resize
        img_resized = cv2.resize(
            img_bgr,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA
        )
        # BGR -> grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return gray

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_points,
            R=self.lbp_radius,
            method=self.lbp_method
        )
        return lbp

    def _compute_lbp_histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Hitung histogram dari LBP image sebagai feature vector
        
        Args:
            lbp_image: LBP image hasil dari local_binary_pattern()
        
        Returns:
            Normalized histogram (numpy array)
        """
        # Untuk method="uniform" dengan 8 points → 10 bins
        # Untuk method="default" → 256 bins
        if self.lbp_method == "uniform":
            n_bins = self.lbp_points + 2
        else:
            n_bins = 2 ** self.lbp_points
        
        hist, _ = np.histogram(
            lbp_image.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True  # Normalize histogram
        )
        return hist.astype(np.float32)

    def extract_from_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: citra BGR (hasil cv2.imread atau frame video)
        return: 1D feature vector (np.ndarray shape (n_features,))
        """
        gray = self._preprocess(img_bgr)
        lbp_image = self._compute_lbp(gray)
        histogram = self._compute_lbp_histogram(lbp_image)
        return histogram

    def extract_from_path(self, img_path: str) -> np.ndarray:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Gagal baca gambar: {img_path}")
        return self.extract_from_array(img_bgr)