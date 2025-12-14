import cv2
import numpy as np

# [paste semua fungsi manual LBP, GLCM, KNN dari atas]
def lbp_manual(gray, radius=1, points=8):
    """LBP circular neighborhood manual implementation"""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Generate sampling offsets (circular neighborhood)
    angles = np.linspace(0, 2*np.pi, points, endpoint=False)
    offsets = [(int(radius*np.cos(a)), int(radius*np.sin(a))) for a in angles]
    
    for i in range(radius, h-radius):
        for j in range(radius, w-radius):
            center = gray[i, j]
            code = 0
            
            for k, (di, dj) in enumerate(offsets):
                # Bilinear interpolation untuk sub-pixel accuracy
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbor = gray[ni, nj]
                else:
                    neighbor = center  # border handling
                    
                code |= (neighbor >= center) << k
            
            lbp[i, j] = code
    return lbp

def glcm_manual(img, distance=1, angle=0, levels=256):
    """Manual GLCM computation"""
    h, w = img.shape
    glcm = np.zeros((levels, levels), dtype=np.float32)
    
    # Direction vectors
    if angle == 0:      # horizontal
        di, dj = 0, distance
    elif angle == 90:   # vertical
        di, dj = distance, 0
    else:
        rad = np.deg2rad(angle)
        di, dj = int(distance*np.sin(rad)), int(distance*np.cos(rad))
    
    for i in range(h):
        for j in range(w):
            i2, j2 = i + di, j + dj
            if 0 <= i2 < h and 0 <= j2 < w:
                p1, p2 = img[i,j], img[i2,j2]
                glcm[p1, p2] += 1
    
    # Normalize
    glcm /= glcm.sum()
    return glcm

def glcm_features(glcm):
    """Haralick features manual computation"""
    i, j = np.ogrid[:glcm.shape[0], :glcm.shape[1]]
    
    # Contrast
    contrast = np.sum(glcm * (i - j)**2)
    
    # Homogeneity
    homogeneity = np.sum(glcm / (1 + (i - j)**2))
    
    # Energy (ASM)
    energy = np.sqrt(np.sum(glcm**2))
    
    # Correlation (simplified)
    mean_i = np.sum(i * glcm.sum(1))
    mean_j = np.sum(j * glcm.sum(0))
    std_i = np.sqrt(np.sum(((i - mean_i)**2) * glcm.sum(1)))
    std_j = np.sqrt(np.sum(((j - mean_j)**2) * glcm.sum(0)))
    correlation = np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j + 1e-6)
    
    return np.array([contrast, correlation, energy, homogeneity])

def knn_manual(X_train, y_train, X_test, k=3):
    """Pure numpy KNN with Euclidean distance"""
    dists = np.sqrt(((X_train[:, np.newaxis] - X_test[np.newaxis, :])**2).sum(-1))
    
    # Get k nearest neighbors
    nearest_idx = np.argpartition(dists, k, axis=0)[:k]
    nearest_labels = y_train[nearest_idx]
    
    # Majority vote
    unique, counts = np.unique(nearest_labels, return_counts=True)
    pred = unique[np.argmax(counts)]
    
    return pred, dists[nearest_idx].flatten()

def extract_features_manual(img_bgr):
    # Preprocess
    gray = cv2.cvtColor(cv2.resize(img_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    
    # LBP manual
    lbp = lbp_manual(gray)
    
    # GLCM manual (flatten LBP ke 1D features)
    glcm = glcm_manual(lbp, distance=1, angle=0)
    features = glcm_features(glcm)
    
    # 1st order stats
    features = np.append(features, [gray.mean(), gray.std()])
    
    return features
