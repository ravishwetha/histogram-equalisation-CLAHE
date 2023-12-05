import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # cv2.split() used to split coloured/multi-channel image into separate single-channel images
    # expensive operation in terms of performance(time) - https://www.geeksforgeeks.org/splitting-and-merging-channels-with-python-opencv/
    
    channels = cv2.split(image)
    equalized_channels = [channel_histogram_equalization(ch) for ch in channels]
    return cv2.merge(equalized_channels)

def channel_histogram_equalization(channel):
    # 1. Compute histogram
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    
    # 2. Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # 3. Normalize the CDF
    # Divide by total number of pixels in the image and multiply by max value
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    
    # 4. Map original values of the channel to new values (from normalized CDF)
    equalized_channel = cdf_normalized[channel]
    
    return equalized_channel

# Read images in color
sample_images = [cv2.imread(f"sample0{i}.jpg", cv2.IMREAD_COLOR) for i in range(1, 9)]
equalized_images = [histogram_equalization(img) for img in sample_images]

# Convert BGR images to RGB for better visualization
sample_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in sample_images]
equalized_images_rgb = [cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB) for eq_img in equalized_images]

# Visualize results
for i, (orig_img, eq_img) in enumerate(zip(sample_images_rgb, equalized_images_rgb), 1):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.imsave(f"sample_color_{i}.png", orig_img)
    plt.title(f"Original image {i}")
    plt.subplot(1, 2, 2)
    plt.imshow(eq_img)
    plt.imsave(f"enhanced_color_{i}.png", eq_img)
    plt.title(f"Equalized image {i}")
    plt.show()
