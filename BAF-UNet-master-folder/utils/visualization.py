# import matplotlib.pyplot as plt


# def show_prediction(image, mask, pred):

#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title('Image')

#     plt.subplot(1, 3, 2)
#     plt.imshow(mask, cmap='gray')
#     plt.title('Ground Truth')

#     plt.subplot(1, 3, 3)
#     plt.imshow(pred, cmap='gray')
#     plt.title('Prediction')

#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

def show_prediction(image, mask, pred, title="Prediction Result"):
    """
    Visualize original image, ground truth, and prediction side by side.
    
    Args:
        image: Original RGB image (numpy array, HWC)
        mask: Ground truth mask (binary)
        pred: Model prediction (binary)
    """
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('Model Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.show()