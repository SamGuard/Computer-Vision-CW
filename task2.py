from cgi import test
from doctest import testfile
from fileinput import filename
from multiprocessing.dummy import Array
from numbers import Number
from tokenize import String
from cv2 import imread, sort
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os
from scipy.ndimage import gaussian_filter

TRAIN_DATA_DIR = "task2/Training/png/"
TEST_DATA_DIR = "task2/TestWithoutRotations/images/"
ANNOTATION_DATA_DIR = "task2/TestWithoutRotations/annotations/"


def loadTrainData(dir: str) -> List[Any]:
    return list(map(lambda fileName: [fileName[4:], cv.imread(dir + fileName, cv.COLOR_BGR2GRAY)],  os.listdir(dir)))


def loadTestData(dir: str) -> List[Any]:
    return sorted(
        list(
            map(lambda fileName: [int(fileName.split("_")[2].split(".")[0]), cv.imread(dir + fileName, cv.COLOR_BGR2GRAY)],
                os.listdir(dir))),
        key=lambda x: x[0])


def loadAnnotations(dir: str):
    textFiles: List[List[int, List[str]]] = list(
        # filter all indexs that are -1
        filter(lambda x: x[1] != -1,
               # Read in all files in dir that end in .txt otherwise return -1
               map(lambda fileName: [-1, -1] if
                   (fileName[-4:] != ".txt")
                   else
                   [int(fileName.split("_")[2].split(".")[0]),
                    open(dir + fileName, "r").readlines()],
                   os.listdir(dir)
                   )
               )
    )

    out: List[List[int, List[str, float, float, float, float]]] = []
    for data in textFiles:
        out.append([data[0]])
        # For each line in a file
        for l in data[1]:
            if(len(l) == 0):
                continue
            parts = l.split(", ")
            iconName: str = parts[0]
            # Points for the icon
            px1: float = float(parts[1][1:])
            py1: float = float(parts[2][:-1])
            px2: float = float(parts[3][1:])
            py2: float = float(parts[4][:-2])
            out[-1].append([iconName, px1, py1, px2, py2])

    return sorted(out, key=lambda x: x[0])


# Given an array of scaled icon images find the best match in the image
def getBestMatchForIcon(img, pyramid):
    bestScale = 1
    bestScore = 1
    bestRes = None
    bestPos = None
    for i in range(len(pyramid) - 1, -1, -1):
        res = cv.matchTemplate(img, pyramid[i], cv.TM_SQDIFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
        closer = min(minVal, maxVal)
        if(closer < bestScore):
            bestScale = i
            bestScore = closer
            bestRes = res
            bestPos = minLoc if (minVal < maxVal) else maxLoc

    '''
    plt.subplot(111), plt.imshow(bestRes)
    plt.scatter(bestPos[0], bestPos[1])
    plt.show()
    print("best scale", bestScale)
    '''

    return [bestScore, bestScale, bestPos]


def main():
    print("Loading data...")
    iconData: List[List] = loadTrainData(TRAIN_DATA_DIR)
    testData: List[List] = loadTestData(TEST_DATA_DIR)
    annotData: List[Any] = loadAnnotations(ANNOTATION_DATA_DIR)

    print("Done")
    print("Train data size:", len(iconData))
    print("Test data size:", len(testData))

    image = testData[0][1]

    '''
    plt.subplot(131), plt.imshow(image, cmap="Greys")
    plt.title('Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(icon, cmap="Greys")
    plt.title('Icon'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''

    iconScores = []
    num = 0
    for _, icon in iconData:
        iconPyramid = [icon]

        for i in range(4):
            iconPyramid.append(cv.pyrDown(iconPyramid[-1]))
            #plt.subplot(141 + i), plt.imshow(imgPyramid[-1], cmap="Greys")
            #plt.subplot(141 + i), plt.imshow(iconPyramid[-1], cmap="Greys")

        iconScores.append([num] + getBestMatchForIcon(image, iconPyramid))
        num += 1

    iconScores = sorted(iconScores, key=lambda x: x[1])
    for x in iconScores:
        print(x)

    for i in range(4):
        print(iconData[iconScores[i][0]][0])


main()
