
import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # 1. Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # 2. Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # 3. Normalize the CDF
    # Divide by total number of pixels in the image and multiply by max value
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    
    # 4. Map original values of the image to new values (from normalized CDF)
    equalized_image = cdf_normalized[image]
    # print(cdf_normalized)
    # print(image)
    
    return equalized_image

# Read images in grayscale
sample_images = [cv2.imread(f"sample0{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, 9)]
equalized_images = [histogram_equalization(img) for img in sample_images]

# Visualize results
for i, (orig_img, eq_img) in enumerate(zip(sample_images, equalized_images), 1):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img, cmap='gray')
    plt.imsave(f"sample_gray_{i}.png", orig_img, cmap='gray')
    plt.title(f"Original image {i}")
    plt.subplot(1, 2, 2)
    plt.imshow(eq_img, cmap='gray')
    plt.imsave(f"enhanced_gray_{i}.png", eq_img, cmap='gray')
    plt.title(f"Equalized image {i}")
    plt.show()
