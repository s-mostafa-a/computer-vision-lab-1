import numpy as np


def _assertion_checks(image, kernel, mode):
    assert len(image.shape) == 2, "The image must be a 2d image!"
    assert len(kernel.shape) == 2, "The kernel must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert kernel.shape[0] % 2 == 1 and kernel.shape[
        1] % 2 == 1, "kernel shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(image, kernel_shape, mode):
    padding_i = kernel_shape[0] // 2
    padding_j = kernel_shape[1] // 2
    if mode == "same":
        image = np.pad(image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, image


def _low_pass_filter_using_broadcast(image: np.array, kernel: np.array, mode: str = 'same'):
    _assertion_checks(image=image, kernel=kernel, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              kernel_shape=kernel.shape, mode=mode)
    shape = tuple([image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j] +
                  list(kernel.shape))
    multi_dim_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_kernel), axis=(2, 3))
    return final_image


def _low_pass_filter_without_broadcast(image: np.array, kernel: np.array, mode: str = 'same'):
    _assertion_checks(image=image, kernel=kernel, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              kernel_shape=kernel.shape, mode=mode)
    shape = (image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j)
    final_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            summation = np.sum(
                image[i - padding_i:i + padding_i + 1, j - padding_j:j + padding_j + 1] * kernel)
            final_image[i - padding_i, j - padding_j] = summation
    return final_image


def gaussian_kernel(shape=(5, 5), cutoff_frequency=1.):
    assert len(shape) == 2, "Kernel must be 2d!"
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1
    ax1 = np.linspace(-(shape[0] - 1) / 2., (shape[0] - 1) / 2., shape[0])
    ax2 = np.linspace(-(shape[1] - 1) / 2., (shape[1] - 1) / 2., shape[1])
    gauss1 = np.exp(-0.5 * np.square(ax1) / np.square(cutoff_frequency))
    gauss2 = np.exp(-0.5 * np.square(ax2) / np.square(cutoff_frequency))
    kernel = np.outer(gauss1, gauss2)
    return kernel / np.sum(kernel)


def box_kernel(shape=(3, 3)):
    kernel = np.ones(shape)
    return kernel / np.sum(kernel)


def hybrid_using_box(to_be_high_passed, to_be_low_passed, shape):
    kernel = box_kernel(shape=shape)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   kernel=kernel)


def hybrid_using_gaussian(to_be_high_passed, to_be_low_passed, cutoff_frequency):
    shape = (cutoff_frequency * 4 + 1, cutoff_frequency * 4 + 1)
    kernel = gaussian_kernel(shape=shape, cutoff_frequency=cutoff_frequency)
    return _hybrid(to_be_high_passed=to_be_high_passed, to_be_low_passed=to_be_low_passed,
                   kernel=kernel)


def _hybrid(to_be_high_passed, to_be_low_passed, kernel):
    high_pass = get_high_pass_image(image=to_be_high_passed, kernel=kernel)
    low_pass = get_low_pass_image(image=to_be_low_passed, kernel=kernel)
    return high_pass + low_pass


def get_high_pass_image(image, kernel):
    result_image = image - get_low_pass_image(image=image, kernel=kernel)
    return result_image


def get_low_pass_image(image, kernel):
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
    return _low_pass_filter_using_broadcast(image=channel, kernel=kernel, mode='same')
