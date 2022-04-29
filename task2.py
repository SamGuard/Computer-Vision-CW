import numpy as np
from scipy.ndimage import gaussian_filter

import cv2 as cv
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os
import random
from scipy import signal, ndimage

from taskslib import check, loadTestData, loadAnnotations, loadTrainData, matchTemplateNCC

USE_ROTATION = False
TRAIN_DATA_DIR = "task2/Training/png/"

IS_MATCH_THRESHOLD = 0.96
PYRAMID_LEVELS = 4
TEST_DATA_DIR = "task2/TestWithoutRotations/images/"
ANNOTATION_DATA_DIR = "task2/TestWithoutRotations/annotations/"

if(USE_ROTATION):
    IS_MATCH_THRESHOLD = 0.85
    TEST_DATA_DIR = "task2/test_data_with_rotations/images/"
    ANNOTATION_DATA_DIR = "task2/test_data_with_rotations/annotations/"


# Given an array of scaled icon images find the best match in the image
def getBestMatchForIcon(img, pyramid):
    bestScale = 0
    bestRot = 0
    bestScore = 0
    bestPos = None
    bestRes = None
    for i in range(len(pyramid) - 1, 0, -1):
        if(USE_ROTATION):
            image_list = [ndimage.rotate(
                pyramid[i], ang, reshape=True, cval=255) for ang in range(0, 360, 15)]
        else:
            image_list = [pyramid[i]]

        rotNum = 0
        for template in image_list:
            res = matchTemplateNCC(img, template)
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
            # print(maxVal)
            minVal = abs(minVal)
            closer = max(minVal, maxVal)
            if(closer > bestScore):
                bestScale = i
                bestScore = closer
                bestPos = minLoc if (minVal > maxVal) else maxLoc
                bestRot = rotNum

            rotNum += 1

    # plt.imshow(bestRes)
    # plt.show()
    return [bestScore, bestScale, bestPos, bestRot]


def getIconsInImage(image, icons, annot):
    iconScores = []
    num = 0
    x = 0
    for name, icon in icons:
        icon = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
        iconPyramid = [icon]
        for i in range(PYRAMID_LEVELS):
            lastI = iconPyramid[-1]
            avgKernel = np.zeros((2, 2))  # np.zeros(lastI.shape)
            downScaled = cv.resize(
                lastI, (lastI.shape[0] // 2, lastI.shape[1] // 2), interpolation=cv.INTER_AREA)
            # plt.imshow(downScaled)
            # plt.show()
            iconPyramid.append(gaussian_filter(downScaled, sigma=0.5))

        iconScores.append([num] + getBestMatchForIcon(image, iconPyramid))
        print("Done icon", name)
        num += 1

    iconScores = sorted(iconScores, key=lambda x: x[1], reverse=True)

    matches = []
    for i in iconScores:
        if(i[1] > IS_MATCH_THRESHOLD):
            print(i[1])
            size = 512 / (2**(i[2]))
            pt1 = i[3]
            pt2 = (int(pt1[0] + size), int(pt1[1] + size))
            matches.append(
                {"iconIndex": i[0], "posTop": i[3], "posBott": pt2, "scale": i[2], "rot": i[3]})
        else:
            break

    return check(matches, icons, annot) + [matches]


def templateMatching(iconData: List[List], testData: List[List], annotData: List[Any]):
    print("Doing template matching.")
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    for i in range(len(testData)):
        image = testData[i][1].copy()
        # print(image.shape)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        [x, y, z, matches] = getIconsInImage(gray, iconData, annotData[i])
        truePos += x
        falsePos += y
        falseNeg += z
        for m in matches:
            pt1 = m["posTop"]
            pt2 = m["posBott"]
            cv.rectangle(image, pt1=pt1, pt2=pt2,
                         color=(0, 255, 0), thickness=3)
            cv.putText(image, iconData[m["iconIndex"]][0][:-4], org=(pt1[0] - 10, pt1[1] - 10),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0), thickness=1)
        cv.imwrite("task2_output/test" + str(i+1)+".jpg", image)

    print("")
    print("")
    print(
        f"Total: Precision {round(100*truePos / (falsePos + truePos))}%  Recall {round(100*truePos / (falseNeg + truePos))}%")


def main():
    print("Loading data...")
    iconData: List[List] = loadTrainData(TRAIN_DATA_DIR)
    testData: List[List] = loadTestData(TEST_DATA_DIR)
    annotData: List[Any] = loadAnnotations(ANNOTATION_DATA_DIR)

    print("Done")
    print("Train data size:", len(iconData))
    print("Test data size:", len(testData))

    '''
    plt.subplot(131), plt.imshow(image, cmap="Greys")
    plt.title('Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(icon, cmap="Greys")
    plt.title('Icon'), plt.xticks([]), plt.yticks([])
    plt.show()
    '''

    templateMatching(iconData, testData, annotData)


main()
