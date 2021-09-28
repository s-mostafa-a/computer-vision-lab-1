import numpy as np

from utils import _get_low_pass_for_each_channel


def my_imfilter(image, filter):
    return get_low_pass_image(image=image, filter=filter)


def get_high_pass_image(image, filter):
    result_image = image - get_low_pass_image(image=image, filter=filter)
    return result_image


def get_low_pass_image(image, filter):
    result_image = np.empty(shape=image.shape, dtype=int)
    if len(image.shape) == 3:
        for ch in range(3):
            result_image[:, :, ch] = _get_low_pass_for_each_channel(channel=image[:, :, ch],
                                                                    filter=filter)
    elif len(image.shape) == 2:
        result_image = _get_low_pass_for_each_channel(channel=image, filter=filter)
    else:
        raise Exception
    return result_image
