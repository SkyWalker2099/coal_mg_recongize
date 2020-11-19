import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.svm import LinearSVC
from CNN.vgg16predict import VGG
from sklearn.externals import joblib
from GLCM import get_feature
import sklearn

vgg = VGG()
vgg.load()

model_path = "model/svm.m"
svc = LinearSVC(max_iter=20000)
svc = joblib.load(model_path)

colors = [[220,20,60],
          [0,0,255],
          [0,255,255],
          [0,255,0],
          [255,215,0],
          [255,0,0],
          [250,250,210],
          [139,69,19],
          [255,0,255],
          [230,230,250],
          [0,128,0],
          [255,255,255],
          [0,0,0]]

def find_cluster(points,h,w):
    eps = min(10,min(h,w)//20)
    min_samples = eps*eps*3
    print(min_samples,eps)
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    # dbscan = DBSCAN(min_samples=200, eps=10)
    clusters = dbscan.fit_predict(points)
    return clusters

def find_stones(img):
    # 找出所有的石头，以点集的方式返回
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])

    limg = cv2.GaussianBlur(gimg, (9, 9), 0)

    simg = cv2.filter2D(limg, ddepth=-1, kernel=kernel)

    ret, thresh = cv2.threshold(simg, 0, 255, cv2.THRESH_OTSU)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    opth = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    clth = cv2.morphologyEx(opth, cv2.MORPH_CLOSE, kernel2)

    points = []
    h,w = clth.shape
    for i in range(h):
        for j in range(w):
            if(clth[i][j] == 255):
                points.append([i,j])

    clusters = find_cluster(points,h,w)
    # for i in range(len(clusters)):
    #     print(clusters[i])
    type_num = max(clusters)
    # print(type_num)

    stones = []
    for type in range(type_num + 1):
        stone = [(points[i][0], points[i][1]) for i in range(len(clusters)) if clusters[i] == type]
        if(len(stone) < h*w/100):
            continue
        stones.append(stone)

    # print(len(stones))
    image = img.copy()
    # image = np.ones(shape=img.shape)
    # image = image*255
    for i in range(len(stones)):
        stone = stones[i]
        for point in stone:
            color = colors[i]
            image[point[0]][point[1]] = color

    # plt.subplot(1, 2, 1)
    # plt.imshow(simg,cmap="gray")
    # plt.subplot(1, 2, 1)
    # plt.imshow(clth,cmap="binary")
    # plt.subplot(1, 2, 2)
    # plt.imshow(image)
    # #
    # plt.show()

    return stones

def centroid(stone):
    x=0
    y=0
    for p in stone:
        x = x + p[1]
        y = y + p[0]
    x/=len(stone)
    y/=len(stone)
    x = int(x)
    y = int(y)
    return (x,y)

def find_boxs(img,stones):
    boxs = []
    boxs2 = []
    centroids = []
    for stone in stones:
        box = []
        minx,miny = img.shape[:2]
        maxx = maxy = 0
        for poi in stone:
            minx = min(poi[0],minx)
            miny = min(poi[1],miny)
            maxx = max(poi[0],maxx)
            maxy = max(poi[1],maxy)
        box = [[miny, minx],
               [maxy, minx],
               [maxy, maxx],
               [miny, maxx]]
        boxs.append(box)

        box2 = cv2.minAreaRect(np.array(stone))
        box2 = cv2.boxPoints(box2)
        box2 = [[box2[1][1],box2[1][0]],
                [box2[0][1],box2[0][0]],
                [box2[3][1],box2[3][0]],
                [box2[2][1],box2[2][0]]]
        boxs2.append(box2)

        c = centroid(stone)
        # print(c)
        centroids.append(c)
    # for box in boxs:
    #     pts = np.array(box, np.int32)
    #     cv2.polylines(img, [pts], True, (255,0,0),thickness=10)

    # for box,c in zip(boxs2,centroids):
    #     pts = np.array(box, np.int32)
    #     cv2.polylines(img, [pts], True, (0,0,255),thickness=10)
    #     cv2.circle(img,center=c,radius=5,color=(255,0,0),thickness=10)
    # plt.imshow(img)
    # plt.show()

    return c,boxs,boxs2

def vgg_predict(inputs):
    inputs1 = []
    for input in inputs:
        inp = cv2.resize(input,(150,150))
        inputs1.append(inp)
    return vgg.predict(inputs1)

def svm_predict(inputs):
    featuress = []
    for input in inputs:
        features = get_feature(input)
        featuress.append(features)
    return svc.predict(featuress)

def set_labels(img,boxs,boxs2):

    inputs = []
    for box in boxs:
        miny,minx = box[0]
        maxy,maxx = box[2]
        image = img[minx:maxx, miny:maxy]
        inputs.append(image)
        # plt.imshow(image)
        # plt.show()
    inputs = np.array(inputs)

    results = vgg_predict(inputs)
    # results = svm_predict(inputs)

    for box,res in zip(boxs2,results):
        pts = np.array(box, np.int32)
        if(res == 1):
            cv2.polylines(img, [pts], True, (255, 0, 0), thickness=10)
        else:
            cv2.polylines(img, [pts], True, (0, 255, 0), thickness=10)
    plt.imshow(img)
    plt.show()

    return img

def pic_cmr(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if (max(h, w) > 1000):
        if (h > w):
            img = cv2.resize(img, dsize=(int(1000 * (w / h)), 1000))
        else:
            img = cv2.resize(img, dsize=(1000, int(1000 * (h / w))))
    stones = find_stones(img.copy())
    c,boxs, boxs2 = find_boxs(img.copy(), stones)
    img_with_label = set_labels(img.copy(), boxs, boxs2)
    return img_with_label,c,boxs2

def cmr(path):
    files = os.listdir(path)
    for file in files:
        pic_cmr(path+file)


