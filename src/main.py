import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import canny


def main():
    image_path, low_threshold_ratio, high_threshold_ratio = sys.argv[1:]
    low_threshold_ratio = float(low_threshold_ratio)
    high_threshold_ratio = float(high_threshold_ratio)
    image = Image.open(image_path)
    gray_image = image.convert('I')
    img = np.array(gray_image)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    img_w_edge = canny.detect(img, low_threshold_ratio=low_threshold_ratio,
                              high_threshold_ratio=high_threshold_ratio)
    plt.subplot(122)
    plt.imshow(img_w_edge, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
