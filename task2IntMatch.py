import math
import signal

import cv2 as cv
import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt


def NormalizeData(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    return (data-mean_data)/(std_data)


def normaliseCC(imageKernel, templete):
    return (1/(imageKernel.size-1)) * np.sum(NormalizeData(imageKernel)* templete)

def test (image, template):
    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Convolve
    ar = np.flipud(np.fliplr(template))

    out = fftconvolve(image, ar.conj(), mode="full")

    volume = np.prod(template.shape)

    image = fftconvolve(np.square(image), a1, mode="full") - \
            np.square(fftconvolve(image, a1, mode="full")) / volume

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    return out

def matchTemplateHome(image, template):
    print(image.shape)
    print(template.shape)
    image = image[0:,0:,1]
    template = template[0:,0:,1]

    _, imageSize = image.shape
    _, size = template.shape

    imageNormalised = NormalizeData(image)
    templateNormalised = NormalizeData(template)
    heatmap = np.zeros((imageSize - size + 1, imageSize - size + 1))
    for x in range(imageSize - size + 1):
        for y in range(imageSize - size + 1):
            heatmap[x][y] = normaliseCC(imageNormalised[x:x + size, y:y + size], templateNormalised)


    return (heatmap)



