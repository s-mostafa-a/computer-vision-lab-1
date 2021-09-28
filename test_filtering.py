from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from my_imfilter import my_imfilter
from utils import gaussian_filter, box_filter


def main():
    # setup
    test_image = np.array(Image.open('./figs/cat.bmp'))
    plt.axis('off')
    plt.imshow(test_image)
    plt.title('original image')
    plt.savefig("./results/test_image.jpeg")
    plt.show()

    # identity filter
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    identity_image = my_imfilter(test_image, identity_filter)
    plt.axis('off')
    plt.imshow(identity_image)
    plt.title('identity image')
    plt.savefig("./results/identity_image.jpeg")
    plt.show()

    # blur
    blur_filter = box_filter(shape=(3, 3))
    blur_image = my_imfilter(test_image, blur_filter)
    plt.axis('off')
    plt.imshow(blur_image)
    plt.title('blur image')
    plt.savefig("./results/blur_image.jpeg")
    plt.show()

    # Large blur
    large_1d_blur_filter = gaussian_filter(shape=(25, 1), cutoff_frequency=10)
    large_blur_image = my_imfilter(test_image, large_1d_blur_filter)
    large_blur_image = my_imfilter(large_blur_image, large_1d_blur_filter.T)
    plt.axis('off')
    plt.imshow(large_blur_image)
    plt.title('large blur image')
    plt.savefig("./results/large_blur_image.jpeg")
    plt.show()

    # Oriented filter (Sobel Operator)
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_image = my_imfilter(test_image, sobel_filter)
    plt.axis('off')
    plt.imshow(sobel_image + 128)
    plt.title('sobel image')
    plt.savefig("./results/sobel_image.jpeg")
    plt.show()

    # High pass filter (Discrete Laplacian)
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_image = my_imfilter(test_image, laplacian_filter)
    plt.axis('off')
    plt.imshow(laplacian_image + 128)
    plt.title('laplacian image')
    plt.savefig("./results/laplacian_image.jpeg")
    plt.show()

    # High pass "filter" alternative
    alternative_high_pass_image = test_image - blur_image
    plt.axis('off')
    plt.imshow(alternative_high_pass_image + 128)
    plt.title('alternative high pass image')
    plt.savefig("./results/alternative_high_pass_image.jpeg")
    plt.show()


if __name__ == '__main__':
    main()
