from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import _low_pass_filter_using_broadcast, _low_pass_filter_without_broadcast, \
    hybrid_using_gaussian
import time


def broadcast_test():
    im = np.array(Image.open('./figs/bird.bmp'))[:, :, 0]
    # im = np.array(
    #     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
    #      [21, 22, 23, 24, 25]])
    plt.axis('off')
    plt.imshow(im)
    plt.show()

    ker = np.ones((3, 3))
    ker = ker / np.sum(ker)

    plt.axis('off')
    start_time = time.time()
    final = _low_pass_filter_using_broadcast(im, ker)
    print(f'broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()

    plt.axis('off')
    start_time = time.time()
    final = _low_pass_filter_without_broadcast(im, ker)
    print(f'without broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()


def hybrid_images_test():
    bottle = np.array(Image.open('figs/bottle.jpg'))
    starship = np.array(Image.open('figs/starship.jpg'))
    result = hybrid_using_gaussian(to_be_high_passed=starship, to_be_low_passed=bottle,
                                   cutoff_frequency=4)
    # result = hybrid_using_box(to_be_high_passed=cat, to_be_low_passed=dog, size=(21, 21))
    plt.axis('off')
    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    hybrid_images_test()
