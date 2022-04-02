"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209088046


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if representation == LOAD_GRAY_SCALE \
        else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img = (img_color - img_color.min()) / (img_color.max() - img_color.min())  # normalization
    return final_img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.gray()
    plt.imshow(img)
    plt.show()
    return None
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
    new_img = np.dot(imgRGB, yiq_from_rgb.T.copy())
    return new_img
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
    mat = np.linalg.inv(yiq_from_rgb)
    new_img = np.dot(imgYIQ, mat.T.copy())
    return new_img
    pass


def show_img(img: np.ndarray):
    plt.gray()  # in case of grayscale image
    plt.imshow(img)
    plt.show()
    pass


""" (Auxiliary function for histogramEqualize and quantizeImage)
    function that helps me to check if an image is RGB
    if it is, it takes the Y channel as the new image (to work with)
    also saves the original image in YIQ representation for later use
    return: (isRGB, yiq_img, imgOrig)
"""


def case_RGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    isRGB = bool(imgOrig.shape[-1] == 3)  # check if the image is RGB image
    if (isRGB):
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = np.copy(imgYIQ[:, :, 0])  # Y channel of the YIQ image
        return True, imgYIQ, imgOrig
    else:
        return False, None, np.copy(imgOrig)
    pass


def back_to_rgb(yiq_img: np.ndarray, y_to_update: np.ndarray) -> np.ndarray:
    yiq_img[:, :, 0] = y_to_update
    rgb_img = transformYIQ2RGB(yiq_img)
    return rgb_img
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :ret
    """
    # display the input image
    show_img(imgOrig)

    isRGB, yiq_img, imgOrig = case_RGB(imgOrig)
    imgOrig = imgOrig * 255
    imgOrig = (np.around(imgOrig)).astype('uint8')  # round & make sure all pixels are integers

    histOrg, bin_edges = np.histogram(imgOrig.flatten(), 256, [0, 255])
    cumsum = histOrg.cumsum()  # cumulative histogram
    # cdf = cumsum * histOrg.max() / cumsum.max()  # normalize to get the cumulative distribution function

    cdf_m = np.ma.masked_equal(cumsum, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # 255 is the max value we want to reach
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # make sure all pixels are integers

    # mapping the pixels
    imgEq = cdf[imgOrig.astype('uint8')]
    histEQ, bin_edges2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    # display the equalized output
    if isRGB:
        imgEq = (imgEq / 255)
        imgEq = back_to_rgb(yiq_img, imgEq)
        show_img(imgEq)
    else:
        show_img(imgEq)

    return imgEq, histOrg, histEQ
    pass



# function to initialized the boundaries

def init_z(nQuant: int) -> np.ndarray:
    size = int(255 / nQuant)  # The initial size given for each interval (fixed - equal division)
    z = np.zeros(nQuant + 1, dtype=int)  # create an empty array representing the boundaries
    for i in range(1, nQuant):
        z[i] = z[i - 1] + size
    z[nQuant] = 255  # always start at 0 and ends at 255
    return z
    pass


def error(imOrig: np.ndarray, new_img: np.ndarray) -> float:
    all_pixels = imOrig.size
    sub = np.subtract(new_img, imOrig)
    pix_sum = np.sum(np.square(sub))
    return np.sqrt(pix_sum) / all_pixels
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    isRGB, yiq_img, imOrig = case_RGB(imOrig)

    if np.amax(imOrig) <= 1:  # so the picture is normalized
        imOrig = imOrig * 255
    imOrig = imOrig.astype('uint8')

    # find image's histogram
    histOrg, bin_edges = np.histogram(imOrig, 256, [0, 255])

    im_shape = imOrig.shape

    z = init_z(nQuant)  # boundaries
    q = np.zeros(nQuant)  # the optimal values for each ‘cell’

    # lists to return
    qImage_list = list()
    error_list = list()

    for i in range(nIter):

        new_img = np.zeros(im_shape)

        for cell in range(len(q)):  # select the values that each of the segments’ intensities will map to
            if cell == len(q) - 1:  # last iteration
                right = z[cell+1] + 1  # to get 255
            else:
                right = z[cell+1]
            cell_range = np.arange(z[cell], right)
            q[cell] = np.average(cell_range, weights=histOrg[z[cell]:right])  # weighted average
            condition = np.logical_and(imOrig >= z[cell], imOrig < right)
            new_img[condition] = q[cell]  # alter the new value

        MSE = error(imOrig / 255.0, new_img / 255.0)
        error_list.append(MSE)
        if isRGB:
            new_img = back_to_rgb(yiq_img, new_img / 255.0)  # save in new_img the image in RGB form
        qImage_list.append(new_img)

        for bound in range(1, len(z)-1):  # move each boundary to be in the middle of two means
            z[bound] = (q[bound-1] + q[bound]) / 2

        if len(error_list) >= 2:
            if np.abs(error_list[-1] - error_list[-2]) <= 0.000001:  # converges
                break

    plt.plot(error_list)
    plt.show()
    return qImage_list, error_list
    pass