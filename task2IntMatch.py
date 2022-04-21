import math

import cv2 as cv
import imutils as imutils
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

image = cv.imread('data2tests/test_image_1.png')
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(image)
template = cv.imread('data2temp/011-trash.png')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
print(template)
w, h = template.shape[::-1]
_,imageSize = imageGray.shape
print(template.shape)
print(imageGray.shape)
_,size = template.shape




x=0
y=0

# GRAY IMAGE
imageAverage = 0
imageSD = 0

for x in range(imageSize):
    for y in range(imageSize):
        imageAverage = imageAverage + imageGray[x][y]

imageAverage = imageAverage / (imageSize * imageSize)

for x in range(imageSize):
    for y in range(imageSize):
        imageSD = imageSD + (imageGray[x][y] - imageAverage) ** 2

imageSD = math.sqrt(imageSD)


imageNormalised = [[0 for x in range(imageSize)] for y in range(imageSize)]

for x in range(size):
    for y in range(size):
        imageNormalised[x][y] = (imageGray[x][y] - imageAverage) / imageSD

imageNormalised = np.array(imageNormalised)

template = gaussian_filter(template, sigma=1)
template = template[0::2,0::2]
print(template.shape)
_, size = template.shape


while(size>31):

    template = gaussian_filter(template, sigma=1)
    template = template[0::2,0::2]

    print(template.shape)
    _, size = template.shape

    templateNormalised = [[0 for x in range(size)] for y in range(size)]

    #TEMPLATE
    templateAverage = 0
    templateSD = 0
    for x in range(size):
        for y in range(size):
            templateAverage = templateAverage + template[x][y]

    templateAverage = templateAverage/(size*size)

    for x in range(size):
        for y in range(size):
            templateSD = templateSD + (template[x][y] - templateAverage)**2

    templateSD = math.sqrt(templateSD)

    for x in range(size):
        for y in range(size):
            templateNormalised[x][y] = (template[x][y]-templateAverage)/templateSD

    heatmap = [[0 for x in range(imageSize - size +1)] for y in range(imageSize - size +1)]

    for x in range(imageSize - size +1):
        for y in range(imageSize - size + 1):
            total = 0

            heatmap[x][y] = (np.dot(np.array(imageNormalised[x:x+size, y:y+size]).flatten(),np.array(templateNormalised).flatten()) +1)*128
            #print(heatmap[x][y])
            #for x2 in range(size):
            #    for y2 in range(size):
            #        total = total + (templateNormalised[x2][y2] * imageNormalised[x+x2][y+y2])
            #heatmap[x][y] = total
        print("done" + str(x))
    print(np.max(heatmap))


    plt.imshow(heatmap, cmap='gray', vmin=0, vmax=255)
    plt.show()


