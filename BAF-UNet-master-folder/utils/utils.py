import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask, title="Result"):
    """
    Simple and clean visualization for binary segmentation
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth / Prediction')
    ax[1].axis('off')
    
    # Optional: Overlay
    overlay = img.copy()
    if len(overlay.shape) == 2:  # grayscale
        overlay = np.stack([overlay, overlay, overlay], axis=-1)
    overlay[mask > 0] = [255, 0, 0]   # Red overlay on lesion
    
    ax[2].imshow(overlay)
    ax[2].set_title('Prediction Overlay (Red)')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()