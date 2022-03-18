import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt

MIN_ANGLE = np.pi / 180


def getAngleBetweenTwoLines(img):
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  edges = cv.Canny(gray,50,150, apertureSize = 3)

  lines = cv.HoughLines(edges,1,np.pi/180,50, lines=2)

  #print(lines)
  [[_, theta0]] = lines[0]
  
  theta1 = 0
  # Get the next line that isn't the same as the current one
  for l in lines[1:]:
    [[_, tempTheta]] = l
    if abs(tempTheta - theta0) > MIN_ANGLE:
      theta1 = tempTheta
      break
  

  print("Angle between lines", abs(round(math.degrees(theta0 - theta1), 2)))

for i in range(1, 11):
  img = cv.imread("data/image" + str(i) + ".png")
  print("Image", i, end=": ")
  getAngleBetweenTwoLines(img)

'''
plt.subplot(131),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gray, cmap="Greys")
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges, cmap="Greys")
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''