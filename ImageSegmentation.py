import cv2
import matplotlib.pyplot as plt
from CGR import *
import os
import numpy as np
import math

# def edge_detect(img):
#     gimg = cv2.GaussianBlur(img,(9,9),0)
#     edges = cv2.Canny(gimg,100,200)
#     return gimg,edges

def findContour(img):  #找出最大轮廓
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.GaussianBlur(imgray, (3, 3), 0)
    ret, thresh = cv2.threshold(gimg, 0, 255, cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gimg, ave, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,6)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxlen = max(len(x) for x in contours)
    ind = max(ind for ind in range(len(contours)) if len(contours[ind]) == maxlen)
    contour = contours[ind]
    # for pairs in contour:
    #     img[pairs[0][1]][pairs[0][0]] = 255, 0, 0
        # print(pairs)

    # plt.subplot(1, 3, 1)
    # plt.imshow(gimg, cmap = "gray")
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(imgray, cmap="gray")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(thresh, cmap="gray")
    #
    # plt.show()
    return thresh,contour

def center(img):
    thresh, contour = findContour(img.copy())
    tolx = toly = 0
    total = 0
    for i in range(150):
        for j in range(150):
            if (thresh[i][j] == 255):
                tolx = tolx + j
                toly = toly + i
                total = total + 1

    cx = int(tolx / total)
    cy = int(toly / total)
    return cx,cy

def textureFind(img):
    thresh, contour = findContour(img.copy())

    # print(contour)

    tolx = toly = 0
    total = 0
    for i in range(150):
        for j in range(150):
            if (thresh[i][j] == 255):
                tolx = tolx + j
                toly = toly + i
                total = total + 1

    cx = int(tolx / total)
    cy = int(toly / total)
    # print(cx,cy)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    area = cv2.contourArea(contour)

    width = math.sqrt(area/(rect[1][1]/rect[1][0]))

    height = width*(rect[1][1]/rect[1][0])



    rect1 = ((cx,cy), (width, height), rect[2])
    # box1 = cv2.boxPoints(rect1)

    # print(rect)
    # print(box,area)
    # pts = np.array(box, np.int32)
    # pts1 = np.array(box1, np.int32)
    # print(pts)
    # cv2.polylines(img,[pts], True, (255,0,0))
    # cv2.polylines(img, [pts1], True, (255, 0, 0))
    # cv2.circle(img, (cx, cy), 2, (255, 0, 0))

    rows = img.shape[0]
    cols = img.shape[1]

    M = cv2.getRotationMatrix2D((cx,cy), rect[2],1) #获得旋转变换矩阵
    dst = cv2.warpAffine(img,M,(cols, rows))

    sampleW = int(rect1[1][0]/2)
    sampleH = int(rect1[1][1]/2)
    sampleImg = dst[cy-sampleH:cy+sampleH,cx-sampleW:cx+sampleW]

    return sampleImg





if __name__ == '__main__':
    pass
    #
    # types = ["train","test"]
    # types2 = ["coal","mg"]
    # types3 = ["with_light", "without_light"]
    # for t1 in types:
    #     for t2 in types2:
    #         for t3 in types3:
    #             path = "data/"+t1+"/"+t2+"/"+t3+"/";
    #             target_path = "data/"+t1+"/"+t2+"/"+t3+"/";
    #             files = os.listdir(path);
    #             for file in files:
    #                 img = cv2.imread(path+file)
    #                 img2 = textureFind(img.copy())
    #                 print(target_path+file)
    #                 cv2.imwrite(target_path+file, img2)




