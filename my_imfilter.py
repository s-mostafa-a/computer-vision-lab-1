from utils import get_low_pass_image


def my_imfilter(image, filter):
    return get_low_pass_image(image=image, kernel=filter)
