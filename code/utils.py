import numpy as np


def _assertion_checks(image, filter, mode):
    assert len(image.shape) == 2, "The image must be a 2d image!"
    assert len(filter.shape) == 2, "The filter must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert filter.shape[0] % 2 == 1 and filter.shape[
        1] % 2 == 1, "filter shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(image, filter_shape, mode):
    padding_i = filter_shape[0] // 2
    padding_j = filter_shape[1] // 2
    if mode == "same":
        image = np.pad(image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, image


def _low_pass_filter_using_broadcast(image: np.array, filter: np.array, mode: str = 'same'):
    _assertion_checks(image=image, filter=filter, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              filter_shape=filter.shape, mode=mode)
    shape = tuple([image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j] +
                  list(filter.shape))
    multi_dim_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_filter = np.expand_dims(np.expand_dims(filter, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_filter), axis=(2, 3))
    return final_image


def _low_pass_filter_without_broadcast(image: np.array, filter: np.array, mode: str = 'same'):
    _assertion_checks(image=image, filter=filter, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              filter_shape=filter.shape, mode=mode)
    shape = (image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j)
    final_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            summation = np.sum(
                image[i - padding_i:i + padding_i + 1, j - padding_j:j + padding_j + 1] * filter)
            final_image[i - padding_i, j - padding_j] = summation
    return final_image


def gaussian_filter(shape=(5, 5), cutoff_frequency=1.):
    assert len(shape) == 2, "filter must be 2d!"
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    ax1 = np.linspace(-(shape[0] - 1) / 2., (shape[0] - 1) / 2., shape[0])
    ax2 = np.linspace(-(shape[1] - 1) / 2., (shape[1] - 1) / 2., shape[1])
    gauss1 = np.exp(-0.5 * np.square(ax1) / np.square(cutoff_frequency))
    gauss2 = np.exp(-0.5 * np.square(ax2) / np.square(cutoff_frequency))
    filter = np.outer(gauss1, gauss2)
    return filter / np.sum(filter)


def box_filter(shape=(3, 3)):
    filter = np.ones(shape)
    return filter / np.sum(filter)


def _get_low_pass_for_each_channel(channel, filter):
    return _low_pass_filter_using_broadcast(image=channel, filter=filter, mode='same')
