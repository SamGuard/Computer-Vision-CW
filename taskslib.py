import numpy as np
from scipy.ndimage import gaussian_filter
from sqlalchemy import false
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os
import random
from scipy import signal
from scipy.signal import fftconvolve


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


def NormalizeData(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    return (data-mean_data)/(std_data)


def normaliseCC(imageKernel, templete):
    return (1/(imageKernel.size-1)) * np.sum(NormalizeData(imageKernel) * templete)


def matchTemplateNCC(image, template):
    # Numerator
    # f-f(line)
    template = template - np.mean(template)
    # g-g(line)
    image = image - np.mean(image)
    # cross correlation
    flippedIcon = np.flipud(np.fliplr(template))
    # create numerator via convolutuion
    imageIcon = fftconvolve(image, flippedIcon, mode="valid")

    # Denominator
    ones = np.ones(template.shape)
    volume = np.prod(template.shape)
    #setup (g-g(line))^2
    image = fftconvolve(np.square(image), ones, mode="valid") - \
        np.square(fftconvolve(image, ones, mode="valid")) / volume
    # multiply by (f-f(line))^2
    template = np.sum(np.square(template))
    # create f(hat)*g(hat)
    ncc = imageIcon / np.sqrt(image * template)
    # remove divide by zero
    ncc[np.where(np.logical_not(np.isfinite(ncc)))] = 0
    return ncc


def matchTemplateHome(image, template):
    print(image.shape)
    print(template.shape)
    image = image[0:, 0:, 1]
    template = template[0:, 0:, 1]

    _, imageSize = image.shape
    _, size = template.shape

    imageNormalised = NormalizeData(image)
    templateNormalised = NormalizeData(template)
    heatmap = np.zeros((imageSize - size + 1, imageSize - size + 1))
    for x in range(imageSize - size + 1):
        for y in range(imageSize - size + 1):
            heatmap[x][y] = normaliseCC(
                imageNormalised[x:x + size, y:y + size], templateNormalised)

    return (heatmap)
