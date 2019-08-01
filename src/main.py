import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import canny


def main():
    image = Image.open('..\\resources\\test_img.jpg')
    gray_image = image.convert('I')
    img = np.array(gray_image)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    img_w_edge = canny.detect(img, 0.03, 0.06, 50)
    plt.subplot(122)
    plt.imshow(img_w_edge, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
