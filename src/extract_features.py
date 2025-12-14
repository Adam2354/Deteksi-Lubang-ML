import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

class FeatureExtractorLBPGLCM:
    def __init__(
        self,
        resize_width: int = 256,
        resize_height: int = 256,
        lbp_radius: int = 1,
        lbp_points: int = 8,
        lbp_method: str = "default",
        glcm_distances = (1,),
        glcm_angles = (0,),
        glcm_levels: int = 256
    ):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_method = lbp_method
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.glcm_levels = glcm_levels

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
        # Hasil LBP biasanya float, tapi kita mau pakai ke GLCM, jadi kita quantize ke range levels
        lbp_norm = cv2.normalize(
            lbp, None, alpha=0, beta=self.glcm_levels - 1,
            norm_type=cv2.NORM_MINMAX
        )
        lbp_uint8 = lbp_norm.astype(np.uint8)
        return lbp_uint8

    def _compute_glcm_features(self, img_gray_or_lbp: np.ndarray) -> np.ndarray:
        # img harus uint8 dengan nilai [0, levels-1]
        glcm = graycomatrix(
            img_gray_or_lbp,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=self.glcm_levels,
            symmetric=True,
            normed=True
        )

        props = {}
        for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
            props[prop_name] = graycoprops(glcm, prop_name)

        # props[prop_name] shape: (len(distances), len(angles))
        # Flatten semua kombinasi distance-angle
        feature_list = []
        for prop_name in ["contrast", "correlation", "energy", "homogeneity"]:
            feature_list.extend(props[prop_name].ravel())

        # Tambahan fitur statistik orde pertama dari citra (mean, std, entropy sederhana)
        img_flat = img_gray_or_lbp.ravel().astype(np.float32)
        mean_val = np.mean(img_flat)
        std_val = np.std(img_flat)

        # entropy kasar pakai histogram
        hist, _ = np.histogram(img_flat, bins=32, range=(0, self.glcm_levels - 1), density=True)
        hist = hist + 1e-12
        entropy_val = -np.sum(hist * np.log2(hist))

        feature_list.extend([mean_val, std_val, entropy_val])

        return np.array(feature_list, dtype=np.float32)

    def extract_from_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: citra BGR (hasil cv2.imread atau frame video)
        return: 1D feature vector (np.ndarray shape (n_features,))
        """
        gray = self._preprocess(img_bgr)
        lbp_img = self._compute_lbp(gray)
        features = self._compute_glcm_features(lbp_img)
        return features

    def extract_from_path(self, img_path: str) -> np.ndarray:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Gagal baca gambar: {img_path}")
        return self.extract_from_array(img_bgr)