import numpy as np
from scipy.ndimage import gaussian_filter
from sqlalchemy import false
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os
import random
from scipy import signal

from taskslib import check, loadTestData, loadAnnotations, loadTrainData, matchTemplateNCC

USE_ROTATION = False
TRAIN_DATA_DIR = "task2/Training/png/"

TEST_DATA_DIR = "task2/TestWithoutRotations/images/"
ANNOTATION_DATA_DIR = "task2/TestWithoutRotations/annotations/"

if(USE_ROTATION):
    TEST_DATA_DIR = "task2/test_data_with_rotations/images/"
    ANNOTATION_DATA_DIR = "task2/test_data_with_rotations/annotations/"


class Match:
    def __init__(self, distance, identIndex, queryIndex):
        self.distance = distance
        self.queryIdx = queryIndex
        self.trainIdx = identIndex

    def __repr__(self) -> str:
        return f"Distance: {self.distance}, Train ID {self.trainIdx}, Query ID {self.queryIdx}"

    def __str__(self):
        return f"Distance: {self.distance}, Train ID {self.trainIdx}, Query ID {self.queryIdx}"


# ident is the image to identify features in
# query is the image to take features from
def bruteForceMatcher(query, ident):
    matches = []
    for iIndex in range(len(ident)):
        i = ident[iIndex]
        bestMatchIndex = -1
        bestMatchScore = -1
        for qIndex in range(len(query)):
            q = query[qIndex]
            d = np.linalg.norm(q - i)

            if(bestMatchIndex == -1 or d < bestMatchScore):
                bestMatchIndex = qIndex
                bestMatchScore = d

        matches.append(Match(bestMatchScore, iIndex, bestMatchIndex))

    return matches


def getIconInImageSurf(baseImage, image, icon, iconName, params: Dict = None):
    surfSize = 742.4164444856908
    distThresh = 0.1401923495148283
    minGoodMatches = 5
    ransacThresh = 5.9894618741018935

    if(params != None):
        surfSize = params["surfSize"]
        distThresh = params["distThresh"]
        minGoodMatches = params["minGoodMatches"]
        ransacThresh = params["ransacThresh"]

    surf = cv.xfeatures2d.SURF_create(surfSize)
    iconKp, iconDes = surf.detectAndCompute(icon, None)
    imgKp, imgDes = surf.detectAndCompute(baseImage, None)

    matches = bruteForceMatcher(iconDes, imgDes)
    matches = sorted(matches, key=lambda x: x.distance)

    goodMatches = []

    for m in matches:
        if(m.distance < distThresh):
            # print(m)
            goodMatches.append(m)
        else:
            break
    # img3 = cv.drawMatches(image,kp1,iconData[6][1],kp2,goodMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()

    if(len(goodMatches) >= minGoodMatches):
        src_pts = np.float32(
            [iconKp[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [imgKp[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacThresh)
        matchesMask = mask.ravel().tolist()
        h, w, _ = image.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                         ).reshape(-1, 1, 2)
        try:
            dst = cv.perspectiveTransform(pts, M)
        except:
            return [None, None, None]
        image = cv.polylines(image, [np.int32(dst)],
                             True, (0, 255, 0), 3, cv.LINE_AA)
        # plt.imshow(image), plt.show()
        # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                   singlePointColor=None,
        #                   matchesMask=matchesMask,  # draw only inliers
        #                   flags=2)
        topX = min(dst, key=lambda x: x[0][0])
        topY = min(dst, key=lambda y: y[0][1])
        bottX = max(dst, key=lambda x: x[0][0])
        bottY = max(dst, key=lambda y: y[0][1])
        topPos = [topX[0][0], topY[0][1]]
        bottPos = [bottX[0][0], bottY[0][1]]
        cv.putText(image, iconName, org=(int(topPos[0]) - 10, int(topPos[1]) - 10),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 0), thickness=1)

        return [topPos, bottPos, image]
    return [None, None, None]


def surfTemplateMatching(iconData: List[List], testData: List[List], annotData: List[Any], params: Dict = None, training: bool = False):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    num = 0
    for baseImage in list(map(lambda x: x[1], testData)):
        matches = []
        drawingImage: cv.Mat = baseImage.copy()

        for i in range(len(iconData)):  # range(6, 7, 1):
            # print(i)
            icon: cv.Mat = iconData[i][1].copy()

            topPos, bottPos, image = getIconInImageSurf(
                baseImage, drawingImage, icon, iconData[i][0][:-4], params)
            if(topPos != None and bottPos != None):
                drawingImage = image
                matches.append(
                    {"iconIndex": i, "posTop": topPos, "posBott": bottPos})

        # plt.imshow(drawingImage)
        # plt.show()

        [x, y, z] = check(matches, iconData, annotData[num],
                          threshold=10, training=training)
        truePos += x
        falsePos += y
        falseNeg += z
        if(training == False):
            cv.imwrite("task3_output/test" + str(num+1)+".jpg", drawingImage)
        num += 1


    if(training == False):
        print("")
        print("")
        print(
            f"Total: Precision {round(100*truePos / (falsePos + truePos))}%  Recall {round(100*truePos / (falseNeg + truePos))}%")
        return
    else:
        print(
            f"Total: Precision {round(100*truePos / (falsePos + truePos))}%  Recall {round(100*truePos / (falseNeg + truePos))}%")
        return truePos + trueNeg - falsePos - falseNeg


def trainSurfParams(iconData, testData, annotData):
    # surf size=742.4164444856908, dist thresh=0.1401923495148283, min good matches=5, ransac=5.9894618741018935
    defParams = {"surfSize": 742.4164444856908, "distThresh": 0.1401923495148283,
                 "minGoodMatches": 5, "ransacThresh": 5.9894618741018935}
    bestParams = [None, defParams.copy()]
    for i in range(100):
        testP = bestParams[1].copy()
        testP["surfSize"] += 10 * (random.random() - 0.5)
        testP["distThresh"] += 0.01 * (random.random() - 0.5)
        testP["minGoodMatches"] += int(2.5 * (random.random() - 0.5))
        testP["ransacThresh"] += 0.75 * (random.random() - 0.5)

        score = surfTemplateMatching(
            iconData, testData, annotData, testP, True)
        print("score", score)
        if(bestParams[0] == None or score > bestParams[0]):
            bestParams[0] = score
            bestParams[1] = testP.copy()
            print(
                f"New best score: surf size={testP['surfSize']}, dist thresh={testP['distThresh']}, min good matches={testP['minGoodMatches']}, ransac={testP['ransacThresh']}"
            )


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

    surfTemplateMatching(iconData, testData, annotData)
    #trainSurfParams(iconData, testData, annotData)


main()
