import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import canny


def main():
    image_path = os.path.join(os.pardir, 'resources', 'test_img.jpg')
    image = Image.open(image_path)
    gray_image = image.convert('I')
    img = np.array(gray_image)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    img_w_edge = canny.detect(img, low_threshold_ratio=0.03,
                              high_threshold_ratio=0.06, weak_pixel_value=50)
    plt.subplot(122)
    plt.imshow(img_w_edge, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
