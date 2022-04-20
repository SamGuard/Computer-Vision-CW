from doctest import testfile
from fileinput import filename
from multiprocessing.dummy import Array
from numbers import Number
from tokenize import String
from cv2 import imread
import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
from typing import Any, List, Set, Dict, Tuple, Optional
import os

TRAIN_DATA_DIR = "task2/Training/png/"
TEST_DATA_DIR = "task2/TestWithoutRotations/images/"
ANNOTATION_DATA_DIR = "task2/TestWithoutRotations/annotations/"


def main():
    pass


def loadImgData(dir: str) -> List[Any]:
    return list(map(lambda fileName: [fileName[4:], cv.imread(dir + fileName)],  os.listdir(dir)))


def loadAnnotations(dir: str):
    textFiles: List[List[str]] = list(
        # filter all indexs that are -1
        filter(lambda x: x != -1,
               # Read in all files in dir that end in .txt otherwise return -1
               map(lambda fileName: -1 if
                   (fileName[-4:] != ".txt")
                   else
                   open(dir + fileName, "r").readlines(), os.listdir(dir)
                   )
               )
    )

    out: List[List[str, float, float, float, float]] = []
    for data in textFiles:
      out.append([])
      #For each line in a file
      for l in data:
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
    return out



print("Loading data...")
trainData: List[List] = loadImgData(TRAIN_DATA_DIR)
testData: List[List] = loadImgData(TEST_DATA_DIR)
annotData: List[Any] = loadAnnotations(ANNOTATION_DATA_DIR)

print("Done")
print("Train data size:", len(trainData))
print("Test data size:", len(testData))
