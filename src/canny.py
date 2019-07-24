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
