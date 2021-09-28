import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from hybrid_test import hybrid_using_gaussian


def get_resized_images(image, scales=5):
    ax1 = image.shape[0]
    ax2 = image.shape[1]
    for i in range(scales):
        resized = np.array(Image.fromarray(image).resize((int(ax2 / (i + 1)), int(ax1 / (i + 1))),
                                                         Image.BILINEAR))
        yield resized


def main():
    cat = np.array(Image.open('./figs/cat.bmp'))
    dog = np.array(Image.open('./figs/dog.bmp'))
    result = hybrid_using_gaussian(to_be_high_passed=cat, to_be_low_passed=dog, cutoff_frequency=5)
    result[result > 255] = 255
    result[result < 0] = 0
    for r in get_resized_images(result.astype(np.uint8)):
        plt.axis('off')
        plt.imshow(r)
        plt.show()


if __name__ == "__main__":
    main()
