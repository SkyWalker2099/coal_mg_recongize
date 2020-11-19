import numpy as np
from skimage.feature import greycomatrix,greycoprops
from skimage import io, color, img_as_ubyte
import pandas as pd
import os
import cv2
bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])

def allin():

    coal_paths = ['data/train_samples/coal/with_light/', 'data/train_samples/coal/without_light/','data/test_samples/coal/with_light/','data/test_samples/coal/without_light/']
    mg_paths = ['data/train_samples/mg/with_light/', 'data/train_samples/mg/without_light/','data/train_samples/mg/too_much_light/',
                'data/test_samples/mg/with_light/', 'data/test_samples/mg/without_light/','data/test_samples/mg/too_much_light/',]
    data_paths = [coal_paths, mg_paths]


    # print(data_paths)
    # return


    count = 0;

    features = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    pandas_features = []
    for feature in features:
        for dis in [1, 2, 4]:
            for dir in range(4):
                f = feature + str(dir) + str(dis)
                pandas_features.append(f)

    pandas_features.append("label")

    for i in range(len(data_paths)):  # 0 表示coal 1 表示 mg
        paths = data_paths[i]
        for path in paths:
            # print(path)
            files = os.listdir(path)
            strs = path.split("/")
            # print(path)
            # csv_name = "csv/glcm/" + path.split("/")[1] +"_"+path.split("/")[2]+ "_" + path.split("/")[3] + ".csv"
            csv_name = "csv/glcm/" + strs[1].split("_")[0]+"/"+strs[2]+"_"+strs[3]+".csv"
            print(csv_name)

            f_list = []

            # continue
            for file in files:
                p = path + file
                p1 = p.split("/")[1] + "_" + p.split("/")[2]
                img = io.imread(p)

                img_feature = get_feature(img)
                img_feature = np.hstack((img_feature,i))
                # print(img_feature)

                f_list.append(img_feature)

                # print(contrast,dissimilarity,homogeneity,energy,correlation,asm,"\n")
                count = count + 1

                # break
            # print(f_list)
            # continue
            mat = np.array(f_list)
            print(mat.shape)
            df = pd.DataFrame(mat, columns=pandas_features)
            print(df)
            print(csv_name)
            df.to_csv(csv_name, index=False)

            # break
            print("*" * 20)
        # break
        print("*" * 20)
    print(count)

def img_2_glcm(img):  #用rgb
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, [1,2,4], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                      levels=max_value, normed=False, symmetric=False)

    # print(matrix_coocurrence)
    # print(matrix_coocurrence.shape)

    return matrix_coocurrence

def get_feature(img):
    matrix_coocurrence = img_2_glcm(img)

    contrast = greycoprops(matrix_coocurrence, "contrast")
    cs = np.hstack((contrast[0], contrast[1], contrast[2]))

    dissimilarity = greycoprops(matrix_coocurrence, "dissimilarity")
    ds = np.hstack((dissimilarity[0], dissimilarity[1], dissimilarity[2]))

    homogeneity = greycoprops(matrix_coocurrence, "homogeneity")
    hs = np.hstack((homogeneity[0], homogeneity[1], homogeneity[2]))

    energy = greycoprops(matrix_coocurrence, "energy")
    es = np.hstack((energy[0], energy[1], energy[2]))

    correlation = greycoprops(matrix_coocurrence, "correlation")
    cos = np.hstack((correlation[0], correlation[1], correlation[2]))

    asm = greycoprops(matrix_coocurrence, "ASM")
    asms = np.hstack((asm[0], asm[1], asm[2]))

    img_feature = np.hstack((cs, ds, hs, es, cos, asms))
    return img_feature

if __name__ == '__main__':
    allin()
    pass