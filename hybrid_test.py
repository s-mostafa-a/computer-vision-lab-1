import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from my_imfilter import get_high_pass_image, get_low_pass_image
from utils import box_filter, gaussian_filter


def hybrid_using_box(to_be_high_passed, to_be_low_passed, shape):
    filter = box_filter(shape=shape)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   filter=filter)


def hybrid_using_gaussian(to_be_high_passed, to_be_low_passed, cutoff_frequency):
    shape = (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1)
    filter = gaussian_filter(shape=shape, cutoff_frequency=cutoff_frequency)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   filter=filter)


def _hybrid(to_be_high_passed, to_be_low_passed, filter):
    high_pass = get_high_pass_image(image=to_be_high_passed, filter=filter)
    low_pass = get_low_pass_image(image=to_be_low_passed, filter=filter)
    return high_pass + low_pass


def hybrid_images_test():


    # my hybrid images
    child = np.array(Image.open('figs/childhood.jpg'))
    teen = np.array(Image.open('figs/teenage.jpg'))
    result = hybrid_using_gaussian(to_be_high_passed=teen, to_be_low_passed=child,
                                   cutoff_frequency=3)
    plt.axis('off')
    plt.title("childhood behind teenage")
    plt.imshow(result)
    plt.savefig("./results/hybrid/child_teen.png")
    plt.show()

    starship = np.array(Image.open('figs/starship.jpg'))
    bottle = np.array(Image.open('figs/bottle.jpg'))
    result = hybrid_using_gaussian(to_be_high_passed=starship, to_be_low_passed=bottle,
                                   cutoff_frequency=5)
    plt.axis('off')
    plt.title("bottle behind starship rocket")
    plt.imshow(result)
    plt.savefig("./results/hybrid/bottle_starship.png")
    plt.show()
    # result = hybrid_using_box(to_be_high_passed=cat, to_be_low_passed=dog, size=(21, 21))


if __name__ == "__main__":
    hybrid_images_test()
