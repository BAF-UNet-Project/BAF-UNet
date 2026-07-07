# import cv2
# import numpy as np


# def get_boundary(mask):

#     mask = mask.astype(np.uint8)

#     kernel = np.ones((3, 3), np.uint8)

#     dilation = cv2.dilate(mask, kernel, iterations=1)
#     erosion = cv2.erode(mask, kernel, iterations=1)

#     boundary = dilation - erosion


# import cv2
# import numpy as np

# def get_boundary(mask: np.ndarray) -> np.ndarray:
#     """
#     Extract boundary map from binary mask
#     """
#     mask = mask.astype(np.uint8)

#     kernel = np.ones((3, 3), np.uint8)

#     dilation = cv2.dilate(mask, kernel, iterations=1)
#     erosion = cv2.erode(mask, kernel, iterations=1)

#     boundary = dilation - erosion  # FIXED (no comma)

#     return boundary.astype(np.float32)





import cv2
import numpy as np

def get_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary map from binary mask using morphological operations.
    
    Args:
        mask: Binary mask (0 and 1) as numpy array
    
    Returns:
        Boundary map (float32) where boundary pixels are 1, others 0
    """
    # Ensure mask is binary and uint8
    mask = (mask > 0).astype(np.uint8)
    
    # Define kernel for dilation/erosion
    kernel = np.ones((3, 3), np.uint8)
    
    # Morphological operations
    dilation = cv2.dilate(mask, kernel, iterations=1)
    erosion = cv2.erode(mask, kernel, iterations=1)
    
    # Boundary = difference between dilation and erosion
    boundary = dilation - erosion
    
    return boundary.astype(np.float32)