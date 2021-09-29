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
    image_pairs_cutoff = [('dog', 'cat', 'bmp', 7),
                          ('einstein', 'marilyn', 'bmp', 4),
                          ('bicycle', 'motorcycle', 'bmp', 9),
                          ('submarine', 'fish', 'bmp', 4),
                          ('bird', 'plane', 'bmp', 6),
                          ('bottle', 'starship', 'jpg', 5),
                          ('childhood', 'teenage', 'jpg', 3)]

    for ipc in image_pairs_cutoff:
        low = np.array(Image.open(f'./figs/{ipc[0]}.{ipc[2]}'))
        high = np.array(Image.open(f'./figs/{ipc[1]}.{ipc[2]}'))
        result = hybrid_using_gaussian(to_be_high_passed=high, to_be_low_passed=low,
                                       cutoff_frequency=ipc[3])
        result[result > 255] = 255
        result[result < 0] = 0
        for r in get_resized_images(result.astype(np.uint8)):
            plt.axis('off')
            plt.title(f'{ipc[0]}_{ipc[1]} {r.shape[0]}x{r.shape[1]}')
            plt.imshow(r)
            plt.savefig(f'./results/hybrid/{ipc[0]}_{ipc[1]}/{r.shape[0]}x{r.shape[1]}.jpg')
            plt.show()


if __name__ == "__main__":
    main()
