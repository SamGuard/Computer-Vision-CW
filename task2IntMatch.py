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

def matchTemplateNCC(image, template):
    #Numerator
    #f-f(line)
    template = template - np.mean(template)
    #g-g(line)
    image = image - np.mean(image)
    #cross correlation
    flippedIcon = np.flipud(np.fliplr(template))
    #create numerator via convolutuion
    imageIcon = fftconvolve(image, flippedIcon, mode="valid")

    #Denominator
    ones = np.ones(template.shape)
    volume = np.prod(template.shape)
    #setup (g-g(line))^2
    image = fftconvolve(np.square(image), ones, mode="valid") - np.square(fftconvolve(image, ones, mode="valid")) / volume

    #multiply by (f-f(line))^2
    template = np.sum(np.square(template))
    #create f(hat)*g(hat)
    ncc = imageIcon / np.sqrt(image * template)
    #remove divide by zero
    ncc[np.where(np.logical_not(np.isfinite(ncc)))] = 0
    return ncc

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



