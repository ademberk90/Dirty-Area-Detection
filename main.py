import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

from numpy.lib.scimath import log

random.seed(12345)

# INPUT // read image as grayscale and resize 
image = cv2.imread("dirty_area2.jpg",0)

# PREPROCESSING // resize image
if image.shape[0]>800 and image.shape[1]>800: 
    image = cv2.resize(image,(800,800))

# PROCESSING // Periodic Pattern Detection -> frequency transform
fourier = np.fft.fft2(image)
fourier_shift = np.fft.fftshift(fourier)
abs_fourier_shift = np.abs(fourier_shift)
log_fourier_shift = np.log(abs_fourier_shift)

plt.subplot(1,4,1)
plt.title("Frequency Domain")
plt.imshow(log_fourier_shift)

# PROCESSING // Periodic Pattern Detection -> Threshold
    # Find the mean value for the thresholding (Its should be close max value)
print(np.min(log_fourier_shift))
print(np.max(log_fourier_shift))
amplitute_threshold = np.max(log_fourier_shift)  * 1
    # Define our new filtered image
periodic_filtered_image = fourier_shift

    # Filter image 
for m in range(image.shape[0]):
    for n in range(image.shape[1]):
        if log_fourier_shift[m][n] > amplitute_threshold:
            periodic_filtered_image[m][n] = complex(0,0)
        else:
            periodic_filtered_image[m][n] = fourier_shift[m][n]

    # Inverse Fourier -> Time domain transform
f_ishift = np.fft.ifftshift(periodic_filtered_image)
periodic_filtered_image = np.fft.ifft2(f_ishift)
periodic_filtered_image = np.abs(periodic_filtered_image)

plt.subplot(1,4,2)
plt.title("Filtered Image")
plt.imshow(periodic_filtered_image)
# plt.plot(fourier_shift)


# PROCESSING // Sobel Edge Detection
    # Take the real parts of image 
filtered_image = np.real(periodic_filtered_image)
    # Convert the dtype 
filtered_image = filtered_image.astype('uint8')
    # Apply Gaussian Blur for remove noise 
filtered_image = cv2.GaussianBlur(filtered_image, (3, 3), 0)

    #Define our Sobel paremetherrs
scale = 1
delta = 0
depth = cv2.CV_16S

    # Convert image to binary, preprocessing for Sobel
ret,binary = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # calculate the derivatives in x and y directions
grad_x = cv2.Sobel(binary, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(binary, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Converting back to CV_8U
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
    # Gradient
sobel_output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

plt.subplot(1,4,3)
plt.title("Sobel Output")
plt.imshow(sobel_output, cmap='gray')

    #SELECTIVE DIRT AREA // Find contours of image
contours, hierarchy = cv2.findContours(sobel_output, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    # Approximate contours to polygons and bounding rects
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
print(len(contours))
for i, c in enumerate(contours):
    # Approximates a polygonal curve
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    # Calculates the up-right bounding rectangle of a point
    boundRect[i] = cv2.boundingRect(contours_poly[i])

    # Define our new image for drawing
drawing = np.zeros((sobel_output.shape[0], sobel_output.shape[1], 3), dtype=np.uint8)

    #  Draw polygonal contour + bonding rects 
for i in range(len(contours)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv2.drawContours(drawing, contours_poly, i, color)
    cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
      (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

plt.subplot(1,4,4)
plt.title("Dirty Area")
plt.imshow(drawing, cmap='gray')
plt.show()
