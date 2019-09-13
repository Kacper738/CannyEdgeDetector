# CannyEdgeDetector
### General info
From-scratch implementation of Canny algorithm used to find edges on image.
### How it works?
Algorithm consists of 5 stages:
1. Gaussian filter to blur image
2. Sobel filter to find gradient's magnitude and slope
3. Non max supression to make edge lines thinner
4. Double thresholding to split edge pixels into weak, strong and non-relevant
5. Hysteresis - changes weak pixels into strong if they have at least one
 strong neighbour
 
 Before applying algorithm image is being converted to 1-channel representation
 as implementation works only with images in this format.
### How to use it?
Download repository, install packages from requirements file and run main.py
with appropriate arguments:

1. image_path
2. low_threshold_ratio - numerical value in 0.0 - 1.0 range
3. high_threshold_ratio - numerical value in 0.0 - 1.0 range