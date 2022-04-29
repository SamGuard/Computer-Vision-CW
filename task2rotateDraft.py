import numpy as np
import pygame as pygame
from scipy.ndimage import gaussian_filter

import cv2 as cv
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os
import random
from scipy import signal, ndimage

from task2IntMatch import matchTemplateNCC

TRAIN_DATA_DIR = "task2/Training/png/"
TEST_DATA_DIR = "task2/test_data_with_rotations/images/"
ANNOTATION_DATA_DIR = "task2/test_data_with_rotations/annotations/"

MATCHING_ALGO = cv.TM_CCORR_NORMED
IS_MATCH_THRESHOLD = 0.80
PYRAMID_LEVELS = 4


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


def check(matches: List[Dict], icons, annot, threshold=0.5, training=False):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    # matches structure:
    # {"iconIndex", "posTop", "posBott"}
    if(not training):
        print("Icons found:")
        for m in matches:
            print(icons[m["iconIndex"]][0][:-4],
                  m["posTop"], m["posBott"], end=", ")

    if(not training):
        print("")
        print("Actual:")
        for a in annot[1:]:
            print(a[0], f"({a[1]}, {a[2]}) ({a[3]}, {a[4]})", end=", ")
        print("")

    doesPass: bool

    # Check all icons found are in the image
    errorFound: str = ""
    doesPass = False
    for m in matches:
        name = icons[m["iconIndex"]][0][:-4]
        x1 = m["posTop"][0]
        y1 = m["posTop"][1]
        x2 = m["posBott"][0]
        y2 = m["posBott"][1]
        doesPass = False
        for a in annot[1:]:
            if(a[0] == name and abs(a[1] - x1) < threshold and abs(a[2] - y1) < threshold and abs(a[3] - x2) < threshold and abs(a[4] - y2) < threshold):
                doesPass = True
                truePos += 1
                break

        if(doesPass == False):
            falsePos += 1

    # Check all icons have been found
    for a in annot[1:]:
        doesPass = False
        name = a[0]
        for m in matches:
            if(name == icons[m["iconIndex"]][0][:-4]):
                doesPass = True
                break

        if(doesPass == False):
            falseNeg += 1
    if(not training):
        print("True Pos, False Pos, False Neg")
        print(f"{truePos}, {falsePos}, {falseNeg}")
        print("")
    return [truePos, falsePos, falseNeg]

def normImage(img):
    avg = np.mean(img)
    return (img - avg) / (sum((img - avg)**2)**0.5)

def matchTemplateHome(image, template):
    image = image[0:,0:,1]
    template = template[0:,0:,1]

    _, size = template.shape
    _, imageSize = image.shape

    imageNormalised = normImage(image)
    templateNormalised = normImage(template)

    heatmap = signal.convolve(imageNormalised,templateNormalised, mode= "valid")

    return (heatmap)
    #return (heatmap / (size ** 2))

# Given an array of scaled icon images find the best match in the image
def getBestMatchForIcon(img, pyramid):
    bestScale = 0
    bestScore = 0
    bestPos = None
    for i in range(len(pyramid) - 1, -1, -1):
        image_list = [ndimage.rotate(pyramid[i], ang, reshape=True, cval=255) for ang in range(0, 360, 45)]
        for o in range(len(image_list)):
            res = matchTemplateNCC(img, image_list[o])
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
            #print(maxVal)
            minVal = abs(minVal)
            closer = max(minVal, maxVal)
            if(closer > bestScore):
                bestScale = i
                bestScore = closer
                bestPos = minLoc if (minVal > maxVal) else maxLoc

    '''
    plt.subplot(111), plt.imshow(bestRes)
    plt.scatter(bestPos[0], bestPos[1])
    plt.show()
    print("best scale", bestScale)
    '''

    return [bestScore, bestScale, bestPos]


def getIconsInImage(image, icons, annot):
    iconScores = []
    num = 0
    x=0
    for _, icon in icons:
        icon = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
        iconPyramid = [icon]

        for i in range(PYRAMID_LEVELS):
            icon = gaussian_filter(icon, sigma=0.9)
            icon = icon[0::2, 0::2]
            iconPyramid.append(icon)
            #iconPyramid.append(cv.pyrDown(iconPyramid[-1]))

        iconScores.append([num] + getBestMatchForIcon(image, iconPyramid))
        num += 1

    iconScores = sorted(iconScores, key=lambda x: x[1], reverse=True)

    matches = []
    for i in iconScores:
        print(i[1])
        if(i[1] > IS_MATCH_THRESHOLD):
            size = 512 / (2**(i[2]))
            pt1 = i[3]
            pt2 = (int(pt1[0] + size), int(pt1[1] + size))
            matches.append(
                {"iconIndex": i[0], "posTop": i[3], "posBott": pt2, "scale": i[2]})
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
        print(image.shape)
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

    s = truePos + falsePos + trueNeg + falsePos

    print("")
    print("")
    print(
        f"Total: True Positive {round(100*truePos / s)}%  False Positive {round(100*falsePos / s)}%    False Negative {round(100*falseNeg / s)}%")


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
    surfSize = 750.0
    distThresh = 0.15
    minGoodMatches = 10.0
    ransacThresh = 5.0

    if(params != None):
        surfSize = params["surfSize"]
        distThresh = params["distThresh"]
        minGoodMatches = params["minGoodMatches"]
        ransacThresh = params["ransacThresh"]

    bf = cv.BFMatcher.create(normType=cv.NORM_L2, crossCheck=True)

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

    s = truePos + falsePos + trueNeg + falseNeg

    if(training == False):
        print("")
        print("")
        print(
            f"Total: True Positive {round(100*truePos / s)}%  False Positive {round(100*falsePos / s)}%    False Negative {round(100*falseNeg / s)}%")
        return
    else:
        print(
            f"Total: True Positive {truePos}%  False Positive {falsePos}%    False Negative {falseNeg}%")
        return truePos + trueNeg - falsePos - falseNeg


def trainSurfParams(iconData, testData, annotData):
    defParams = {"surfSize": 736.0849596093133, "distThresh": 0.1530496234754787,
                 "minGoodMatches": 8, "ransacThresh": 5.58664439503451}
    bestParams = [None, defParams.copy()]
    for i in range(100):
        testP = bestParams[1].copy()
        testP["surfSize"] += 20 * (random.random() - 0.5)
        testP["distThresh"] += 0.01 * (random.random() - 0.5)
        testP["minGoodMatches"] += int(2.5 * (random.random() - 0.5))
        testP["ransacThresh"] += 1 * (random.random() - 0.5)

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

    templateMatching(iconData, testData, annotData)

    #surfTemplateMatching(iconData, testData, annotData)
    #trainSurfParams(iconData, testData, annotData)


main()
