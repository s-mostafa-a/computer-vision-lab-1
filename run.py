from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import low_pass_filter_using_broadcast, low_pass_filter_without_broadcast, \
    gaussian_kernel, box_kernel
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
    final = low_pass_filter_using_broadcast(im, ker)
    print(f'broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()

    plt.axis('off')
    start_time = time.time()
    final = low_pass_filter_without_broadcast(im, ker)
    print(f'without broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()


def hybrid_images_test():
    bottle = np.array(Image.open('figs/bottle.jpg'))
    starship = np.array(Image.open('figs/starship.jpg'))
    result = hybrid_using_gaussian(to_be_high_passed=starship, to_be_low_passed=bottle,
                                   cutoff_frequency=4)
    # result = hybrid_using_box(to_be_high_passed=cat, to_be_low_passed=dog, size=21)
    plt.axis('off')
    plt.imshow(result)
    plt.show()


def hybrid_using_box(to_be_high_passed, to_be_low_passed, size):
    kernel = box_kernel(size=size)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   kernel=kernel)


def hybrid_using_gaussian(to_be_high_passed, to_be_low_passed, cutoff_frequency):
    kernel = gaussian_kernel(cutoff_frequency=cutoff_frequency)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   kernel=kernel)


def _hybrid(to_be_high_passed, to_be_low_passed, kernel):
    high_pass = _get_high_pass_image(image=to_be_high_passed, kernel=kernel)
    low_pass = _get_low_pass_image(image=to_be_low_passed, kernel=kernel)
    return high_pass + low_pass


def _get_high_pass_image(image, kernel):
    result_image = image - _get_low_pass_image(image=image, kernel=kernel)
    return result_image


def _get_low_pass_image(image, kernel):
    result_image = np.empty(shape=image.shape, dtype=int)
    if len(image.shape) == 3:
        for ch in range(3):
            result_image[:, :, ch] = _get_low_pass_for_each_channel(channel=image[:, :, ch],
                                                                    kernel=kernel)
    elif len(image.shape) == 2:
        result_image = _get_low_pass_for_each_channel(channel=image, kernel=kernel)
    else:
        raise Exception
    return result_image


def _get_low_pass_for_each_channel(channel, kernel):
    return low_pass_filter_using_broadcast(image=channel, kernel=kernel, mode='same')


if __name__ == "__main__":
    hybrid_images_test()
