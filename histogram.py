# Imported necessary libraries for image processing and visualization
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Function to plot two images side by side for comparison
def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()

# Function to calculate and display histograms of two images side by side
def plot_hist(old_image, new_image, title_old="Original", title_new="New Image"):
    intensity_values = np.array([x for x in range(256)])  # Intensity values from 0 to 255
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_old)
    plt.xlabel('Intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_new)
    plt.xlabel('Intensity')
    plt.show()

# Toy Image Example
# Creating a small 2D array with intensity values
toy_image = np.array([[0, 2, 2], [1, 1, 1], [1, 1, 2]], dtype=np.uint8)

# Displaying the toy image
plt.imshow(toy_image, cmap="gray")
plt.show()
print("Toy image:", toy_image)

# Displaying intensity values of pixels in the toy image
plt.bar([x for x in range(6)], [1, 5, 2, 0, 0, 0])  # Example histogram 1
plt.show()
plt.bar([x for x in range(6)], [0, 1, 0, 5, 0, 2])  # Example histogram 2
plt.show()

# Goldhill Image Example
# Reading an example grayscale image
goldhill = cv2.imread("goldhill.bmp", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(goldhill, cmap="gray")
plt.show()

# Calculating and displaying the histogram of the Goldhill image
hist = cv2.calcHist([goldhill], [0], None, [256], [0, 256])
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title("Bar Histogram")
plt.show()

plt.plot(intensity_values, hist)  # Line plot of the histogram
plt.title("Histogram")
plt.show()

# Baboon Image Example
# Reading a color image and displaying it
baboon = cv2.imread("baboon.png")
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct color display
plt.show()

# Plotting histograms for each color channel (blue, green, red)
color = ('blue', 'green', 'red')
for i, col in enumerate(color):
    histr = cv2.calcHist([baboon], [i], None, [256], [0, 256])
    plt.plot(intensity_values, histr, color=col, label=col + " channel")
    plt.xlim([0, 256])  # Limit x-axis to valid intensity values
plt.legend()
plt.title("Histogram Channels")
plt.show()

# Thresholding Function
# This function applies a binary threshold to an image
def thresholding(input_img, threshold, max_value=255, min_value=0):
    N, M = input_img.shape  # Dimensions of the image
    image_out = np.zeros((N, M), dtype=np.uint8)  # Initialize output image
    for i in range(N):
        for j in range(M):
            if input_img[i, j] > threshold:  # Pixel intensity check
                image_out[i, j] = max_value
            else:
                image_out[i, j] = min_value
    return image_out

# Thresholding Example with Cameraman Image
# Reading a grayscale image
image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.show()

# Calculating and plotting histogram of the original image
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.title("Bar Histogram")
plt.show()

# Applying thresholding to the image
threshold = 87  # Threshold value
max_value = 255  # Maximum intensity for thresholded pixels
min_value = 0  # Minimum intensity for thresholded pixels
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)

# Plotting original and thresholded images side by side
plot_image(image, new_image, "Original", "Image after Threshold")

# Plotting histograms of original and thresholded images
plt.figure(figsize=(10, 10))
plot_hist(image, new_image, "Original", "Image after Threshold")

