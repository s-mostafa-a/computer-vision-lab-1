from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import _low_pass_filter_using_broadcast, _low_pass_filter_without_broadcast, box_filter
import time


def broadcast_test():
    im = np.array(Image.open('./figs/bird.bmp'))[:, :, 0]
    # im = np.array(
    #     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
    #      [21, 22, 23, 24, 25]])
    plt.axis('off')
    plt.imshow(im)
    plt.show()

    fil = box_filter((3, 3))
    fil = fil / np.sum(fil)

    plt.axis('off')
    start_time = time.time()
    final = _low_pass_filter_using_broadcast(im, fil)
    print(f'broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()

    plt.axis('off')
    start_time = time.time()
    final = _low_pass_filter_without_broadcast(im, fil)
    print(f'without broadcast {time.time() - start_time} seconds')
    plt.imshow(final)
    plt.show()


if __name__ == "__main__":
    broadcast_test()
