import numpy as np
from scipy.ndimage.filters import convolve


def gaussian_kernel(size=5, sigma=1.0):
    """
    Returns 2d gaussian kernel

    :param size: int, size of kernel (default 5)
    :param sigma: float, standard deviation (default 1.0)
    :return: ndarray, gaussian kernel
    """
    x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
    normal = 1 / (2 * np.pi * sigma**2)
    kernel = normal * np.exp(-((x**2 + y**2)/(2 * sigma**2)))

    return kernel


def gaussian_filter(image, kernel_size=5, kernel_sigma=1.0):
    """
    Returns image convolved with gaussian kernel

    :param image: 2d array, image
    :param kernel_size: int, size of kernel (default 5)
    :param kernel_sigma: float, standard deviation (default 1.0)
    :return: ndarray, convolved image
    """

    return convolve(image, gaussian_kernel(kernel_size, kernel_sigma))


def non_max_supression(image, theta):
    """
    Finds thinner edges

    :param image: 2d array, image with edges
    :param theta: 2d array, gradient slope
    :return: ndarray, image with thin edges
    """
    height, width = image.shape
    output_image = np.zeros((height, width))
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height-1):
        for j in range(1, width-1):

            # 0 degrees
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                left = image[i, j-1]
                right = image[i, j+1]

            # 45 degrees
            elif 22.5 <= angle[i, j] < 67.5:
                left = image[i-1, j+1]
                right = image[i+1, j-1]

            # 90 degrees
            elif 67.5 <= angle[i, j] < 112.5:
                left = image[i-1, j]
                right = image[i+1, j]

            # 135 degrees
            elif 112.5 <= angle[i, j] < 157.5:
                left = image[i-1, j-1]
                right = image[i+1, j+1]

            if (image[i, j] > left) and (image[i, j] > right):
                output_image[i, j] = image[i, j]

    return output_image


def sobel_filter(image):
    """
    Uses sobel's kernel to calculate image gradient, magnitude and slope
    of this gradient

    :param image: 2d array, image
    :return: 2d array, 2d array, magnitude and slope
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)

    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)

    return G, theta
