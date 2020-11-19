import matplotlib
import sklearn
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import cv2
import os
# import tensorflow

from ImageSegmentation import *
from GLCM import *

coal_data = "data/coal/"
mg_data = "data/mg/"

def CGR(path):

    model_path = "model/svm.m"
    img = cv2.imread(path)
    img = textureFind(img.copy())
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    features = get_feature(img)
    # print(features)
    svc = joblib.load(model_path)
    return svc.predict([features])

if __name__ == '__main__':

    count = 0
    correct = 0

    coal_paths = ["coal/with_light/", "coal/without_light/"]
    mg_paths = ["mg/with_light/", "mg/without_light/","mg/too_much_light/"]
    data_paths = [coal_paths, mg_paths]
    save_path = "data/test/"

    for i in range(len(data_paths)):
        paths = data_paths[i]
        for path in paths:
            files = os.listdir(save_path + path)
            for file in files:
                # try:
                result = CGR(save_path + path + file)
                if result == i:
                    correct = correct + 1
                else:
                    print(path + file)
                count = count + 1
                # except Exception as e:
                    # print(path+file)


    # paths = ["coal_with_light/","mg_with_light/"]
    # t_path = "data/tough_guys/"
    # for i in range(2):
    #     path = paths[i]
    #     files = os.listdir(t_path + path)
    #     for file in files:
    #         try:
    #             img = cv2.imread(t_path + path + file)
    #             result = CGR(t_path + path + file)
    #
    #             if (result == i):
    #                 correct += 1
    #             else:
    #                 plt.title(path + file)
    #                 plt.imshow(img)
    #                 plt.show()
    #
    #             count += 1
    #         except Exception as e:
    #             print(path+file)


    # path = "data/mg/too_much_light/"
    # files = os.listdir(path)
    # for file in files:
    #     try:
    #         img = cv2.imread(path+file)
    #         result = CGR(path+file)
    #         if(result == 1):
    #             correct+=1
    #             print(file)
    #         # else:
    #         #     plt.title(file)
    #         #     plt.imshow(img)
    #         #     plt.show()
    #         count+=1
    #     except Exception as e:
    #         print(file)
    # 以上为测试用例

    print("共有：",count)
    print("正确:",correct)

    pass







