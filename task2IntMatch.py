import math
import signal

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def matchTemplateHome(image, template):
    image = image[0:,0:,1]
    template = template[0:,0:,1]

    _, size = template.shape
    _, imageSize = image.shape

    #TEMPLATE Norm
    templateAverage = np.mean(template)
    templateSD = np.std(template)

    templateNormalised = [[(template[x][y] - templateAverage) / templateSD for x in range(size)] for y in range(size)]

    #IMAGE Norm
    imageAverage = np.mean(image)
    imageSD = np.std(image)

    imageNormalised = [[(image[x][y] - imageAverage) / imageSD for x in range(imageSize)] for y in range(imageSize)]

    #Convolve???? I have tried multiple ways of achieving f hat * g hat from lectures with no sucsess
    heatmap = signal.convolve2d(imageNormalised,templateNormalised, mode= "valid")

    #for x in range(imageSize - size +1):
     #   for y in range(imageSize - size + 1):
      #      heatmap[x][y] = (np.dot(np.array(imageNormalised[x:x+size, y:y+size]).flatten(),np.array(templateNormalised).flatten()) +1)/2

    return (heatmap)
    #return (heatmap / (size ** 2))



