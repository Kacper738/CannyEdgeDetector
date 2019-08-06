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
### How to use?
Image should be in one-channel representation, so before applying algorithm
it needs to be converted (for example function convert from Pillow.Image).
Applyling - use detect function giving image, low_threshold_ratio and
high_threshold_ratio as parameters.