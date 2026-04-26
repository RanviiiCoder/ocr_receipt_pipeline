import cv2
import numpy as np

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detects and corrects skew in an image using minimum-area-rect angle detection.
    """
    # Threshold the image to find the text blocks
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle for rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Applies a sequence of preprocessing steps to an image:
    1. Loads the image
    2. Converts to grayscale
    3. Denoises using fastNlMeansDenoising
    4. Applies adaptive Gaussian thresholding
    5. Corrects skew
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Apply Adaptive Gaussian Thresholding to fix contrast and lighting
    # We use cv2.THRESH_BINARY_INV so that text becomes white on a black background for deskewing
    thresh = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Correct skew
    deskewed_thresh = deskew(thresh)
    
    # Invert back so text is black on white for OCR
    final_img = cv2.bitwise_not(deskewed_thresh)
    
    return final_img
