from cv2 import imread, imshow, imwrite, perspectiveTransform, sort, threshold
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os

TRAIN_DATA_DIR = "task2/Training/png/"
TEST_DATA_DIR = "task2/TestWithoutRotations/images/"
ANNOTATION_DATA_DIR = "task2/TestWithoutRotations/annotations/"

MATCHING_ALGO = cv.TM_CCORR_NORMED
IS_MATCH_THRESHOLD = 0.992
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


def check(matches: List[Dict], icons, annot, threshold=0.5):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    # matches structure:
    # {"iconIndex", "posTop", "posBott"}
    print("Icons found:")
    for m in matches:
        print(icons[m["iconIndex"]][0][:-4],
              m["posTop"], m["posBott"], end=", ")

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

    print("True Pos, False Pos, False Neg")
    print(f"{truePos}, {falsePos}, {falseNeg}")

    print("")
    return [truePos, falsePos, falseNeg]


# Given an array of scaled icon images find the best match in the image
def getBestMatchForIcon(img, pyramid):
    bestScale = 0
    bestScore = 0
    bestPos = None
    for i in range(len(pyramid) - 1, -1, -1):
        res = cv.matchTemplate(img, pyramid[i], MATCHING_ALGO)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
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
    for _, icon in icons:
        iconPyramid = [icon]

        for i in range(PYRAMID_LEVELS):
            iconPyramid.append(cv.pyrDown(iconPyramid[-1]))

        iconScores.append([num] + getBestMatchForIcon(image, iconPyramid))
        num += 1

    iconScores = sorted(iconScores, key=lambda x: x[1], reverse=True)

    matches = []
    for i in iconScores:
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
        [x, y, z, matches] = getIconsInImage(image, iconData, annotData[i])
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
        imwrite("task2_output/test" + str(i+1)+".jpg", image)

    s = truePos + falsePos + trueNeg + falsePos

    print("")
    print("")
    print(
        f"Total: True Positive {round(100*truePos / s)}%  False Positive {round(100*falsePos / s)}%    False Negative {round(100*falseNeg / s)}%")


def getIconInImageSurf(baseImage, image, icon, iconName):
    bf = cv.BFMatcher.create(normType=cv.NORM_L2, crossCheck=True)

    surf = cv.xfeatures2d.SURF_create(100)
    iconKp, iconDes = surf.detectAndCompute(icon, None)
    imgKp, imgDes = surf.detectAndCompute(baseImage, None)

    matches = bf.match(iconDes, imgDes)

    matches = sorted(matches, key=lambda x: x.distance)

    goodMatches = []

    for m in matches:
        if(m.distance < 0.2):
            goodMatches.append(m)

    #img3 = cv.drawMatches(image,kp1,iconData[6][1],kp2,goodMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3)
    # plt.show()

    if(len(goodMatches) >= 10):
        src_pts = np.float32(
            [iconKp[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [imgKp[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, _ = image.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                         ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        image = cv.polylines(image, [np.int32(dst)],
                             True, (0, 255, 0), 3, cv.LINE_AA)
        #plt.imshow(image), plt.show()
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


def surfTemplateMatching(iconData: List[List], testData: List[List], annotData: List[Any]):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    num = 0
    for baseImage in list(map(lambda x: x[1], testData)):
        matches = []
        drawingImage: cv.Mat = baseImage.copy()

        for i in range(len(iconData)):
            icon: cv.Mat = iconData[i][1].copy()

            topPos, bottPos, image = getIconInImageSurf(
                baseImage, drawingImage, icon, iconData[i][0][:-4])
            if(topPos != None and bottPos != None):
                drawingImage = image
                matches.append(
                    {"iconIndex": i, "posTop": topPos, "posBott": bottPos})

        # plt.imshow(drawingImage)
        # plt.show()

        [x, y, z] = check(matches, iconData, annotData[num], threshold=10)
        truePos += x
        falsePos += y
        falseNeg += z
        imwrite("task3_output/test" + str(num+1)+".jpg", drawingImage)
        num += 1

    s = truePos + falsePos + trueNeg + falseNeg

    print("")
    print("")
    print(
        f"Total: True Positive {round(100*truePos / s)}%  False Positive {round(100*falsePos / s)}%    False Negative {round(100*falseNeg / s)}%")


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

    #templateMatching(iconData, testData, annotData)

    surfTemplateMatching(iconData, testData, annotData)


main()
