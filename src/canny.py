import numpy as np
from scipy.ndimage.filters import convolve


def detect(image, low_threshold_ratio, high_threshold_ratio,
           weak_pixel_value=1, strong_pixel_value=255, gaussian_kernel_size=5,
           gaussian_kernel_sigma=1.0):
    """
    Applies canny edge detection algorithm to given image

    :param image: 2d array, image
    :param low_threshold_ratio: float, range: 0.0 to 1.0, low threshold
    :param high_threshold_ratio: float, range: 0.0 to 1.0, high threshold
    :param weak_pixel_value: int, range: 0 to 255, weak pixel value (default 1)
    :param strong_pixel_value: int, range 0 to 255, strong pixel value
    (default 255)
    :param gaussian_kernel_size: int, size of gaussian filter kernel
    (default 5)
    :param gaussian_kernel_sigma: float, standard deviation (default 1.0)
    :return: 2d ndarray, image with edges
    """
    image = gaussian_filter(image, gaussian_kernel_size, gaussian_kernel_sigma)
    image, theta = sobel_filter(image)
    image = non_max_supression(image, theta)
    image = double_threshold(image, low_threshold_ratio, high_threshold_ratio,
                             weak_pixel_value, strong_pixel_value)
    image = hysteresis(image, weak_pixel_value, strong_pixel_value)

    return image


def double_threshold(image, low_threshold_ratio, high_threshold_ratio,
                     weak_pixel_value=1, strong_pixel_value=255):
    """
    Splits pixels from image into three groups by two thresholds

    :param image: 2d array, image
    :param low_threshold_ratio: float, range: 0.0 to 1.0, low threshold
    :param high_threshold_ratio: float, range: 0.0 to 1.0, high threshold
    :param weak_pixel_value: int, range: 0 to 255, weak pixel value
    (default 1)
    :param strong_pixel_value: int, range 0 to 255, strong pixel value
    (default 255)
    :return: 2d ndarray, pixels from original image split into 3 groups
    """
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = image.max() * low_threshold_ratio
    result = np.zeros(image.shape)

    strong_x, strong_y = np.where(image > high_threshold)
    weak_x, weak_y = np.where(np.logical_and(image > low_threshold,
                                             image < high_threshold))

    result[strong_x, strong_y] = strong_pixel_value
    result[weak_x, weak_y] = weak_pixel_value

    return result


def gaussian_kernel(size=5, sigma=1.0):
    """
    Creates 2d gaussian kernel

    :param size: int, size of kernel (default 5)
    :param sigma: float, standard deviation (default 1.0)
    :return: 2d ndarray, gaussian kernel
    """
    x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
    normal = 1 / (2 * np.pi * sigma**2)
    kernel = normal * np.exp(-((x**2 + y**2)/(2 * sigma**2)))

    return kernel


def gaussian_filter(image, kernel_size=5, kernel_sigma=1.0):
    """
    Convolves image with gaussian kernel

    :param image: 2d array, image
    :param kernel_size: int, size of kernel (default 5)
    :param kernel_sigma: float, standard deviation (default 1.0)
    :return: 2d ndarray, convolved image
    """

    return convolve(image, gaussian_kernel(kernel_size, kernel_sigma))


def hysteresis(image, weak_pixel_value=1, strong_pixel_value=255):
    """
    Transforms weak pixels into strong if they have at least one strong
    neighbour

    :param image: 2d array, image
    :param weak_pixel_value: int, value for weak pixel (default 1)
    :param strong_pixel_value: int, value for strong pixel (default 255)
    :return: 2d ndarray, image with only strong or non-relevant pixels
    """
    height, width = image.shape

    for i in range(1, height-1):
        for j in range(1, width-1):

            if image[i, j] == weak_pixel_value:
                if ((image[i-1, j-1] == strong_pixel_value)
                        or (image[i-1, j] == strong_pixel_value)
                        or (image[i-1, j+1] == strong_pixel_value)
                        or (image[i, j-1] == strong_pixel_value)
                        or (image[i, j+1] == strong_pixel_value)
                        or (image[i+1, j-1] == strong_pixel_value)
                        or (image[i+1, j] == strong_pixel_value)
                        or (image[i+1, j+1] == strong_pixel_value)):
                    image[i, j] = strong_pixel_value
                else:
                    image[i, j] = 0

    return image


def non_max_supression(image, theta):
    """
    Finds thinner edges

    :param image: 2d array, image with edges
    :param theta: 2d array, gradient slope
    :return: 2d ndarray, image with thin edges
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
    :return: 2d ndarray, 2d ndarray, magnitude and slope
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)

    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx)
    
    return G, theta
