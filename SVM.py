# import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path1 = "coal_with_light.csv"
path2 = "mg_with_light.csv"
path3 = "coal_without_light.csv"
path4 = "mg_without_light.csv"
path5 = "mg_too_much_light.csv"

origin = "csv/glcm/"
n_features = 72

def train_data(type):
    data1 = pd.read_csv(origin+type+"/"+path1)
    data2 = pd.read_csv(origin+type+"/"+path2)
    data3 = pd.read_csv(origin+type+"/"+path3)
    data4 = pd.read_csv(origin+type+"/"+path4)
    data5 = pd.read_csv(origin+type+"/"+path5)
    csvdata = pd.concat([data1, data2, data3, data4])

    print(csvdata.shape)

    # print(csvdata)
    data = csvdata.values
    # print(data.shape)
    data = np.split(data, [72], axis=1)

    X = data[0]
    Y = data[1]
    Y = Y.ravel()
    return X,Y;


def SVM():

    x_train, y_train = train_data("train")

    x_test, y_test = train_data("test")

    linear_svm = LinearSVC(max_iter=20000).fit(x_train,y_train)
    # linear_svm = LinearSVC()
    # linear_svm = joblib.load("model/svm.m")


    print(linear_svm.score(x_train,y_train))
    print(linear_svm.score(x_test, y_test))

    joblib.dump(linear_svm, "model/svm.m")


def test_SVM():
    x_train, y_train = train_data("train")
    x_test, y_test = train_data("test")
    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train_transed = scaler.transform(x_train)
    x_test_transed = scaler.transform(x_test)

    kernels = ["linear","rbf","poly"]
    gammas = [10/n_features]
    Cs = [0.1,1,10,100,1000,5000]

    for kernel in kernels:
        for gamma in gammas:
            for C in Cs:
                svc = SVC(kernel=kernel, gamma=gamma, C=C, max_iter=20000)
                svc.fit(x_train_transed, y_train)
                print(kernel,gamma,C,svc.score(x_test_transed, y_test))

                # svc = SVC(kernel=kernel, gamma=gamma, C=C, max_iter=20000)
                # svc.fit(x_train, y_train)
                # print(kernel, gamma, C, svc.score(x_test, y_test))




def RFC():

    x_train, y_train = train_data("train")

    x_test, y_test = train_data("test")

    forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)

    print(forest.score(x_train, y_train))
    print(forest.score(x_test, y_test))

    joblib.dump(forest,"model/forest.m")

def test_RFC():
    estimators = range(1,41)
    max_fs = [int(n_features/2), int(math.sqrt(72)), int(n_features)]

    x_train, y_train = train_data("train")
    x_test, y_test = train_data("test")
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_transed = scaler.transform(x_train)
    x_test_transed = scaler.transform(x_test)


    s1=[]
    s2=[]
    s3=[]
    s = [s1,s2,s3]
    for i in range(3):
        max_f = max_fs[i]
        for n_e in estimators:
            forest = RandomForestClassifier(n_estimators=n_e, max_features=max_f).fit(x_train_transed,y_train)
            s[i].append(forest.score(x_test_transed,y_test))
            print(s[i])

    # plt.plot(x, train_accuracys, label="train_acc", color="r")
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # plt.title("loss")
    # plt.legend()
    # plt.show()
    plt.plot(estimators,s1,label = "n_features/2",color="r")
    plt.plot(estimators, s2, label="sqrt(n_features)", color="g")
    plt.plot(estimators, s3, label="n_features", color="b")
    plt.ylabel("score")
    plt.xlabel("n_estimators")
    plt.legend()
    plt.show()

def rfc_feature_importance():
    features = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    pandas_features = []
    for feature in features:
        for dis in [1, 2, 4]:
            for dir in range(4):
                f = feature + str(dir) + str(dis)
                pandas_features.append(f)
    x_train, y_train = train_data("train")
    x_test, y_test = train_data("test")
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_transed = scaler.transform(x_train)
    x_test_transed = scaler.transform(x_test)
    rfc = RandomForestClassifier(n_estimators=100, random_state=0).fit(x_train_transed, y_train)
    feat_importances = rfc.feature_importances_

    # print(feat_importances)
    # max_importance = feat_importances
    plt.figure(dpi=100)
    plt.bar(pandas_features,feat_importances)
    plt.xticks(rotation=270,fontsize=5)
    # plt.bar(bottom=pandas_features,x=feat_importances,height=0.1,orientation="horizontal")
    plt.show()

if __name__ == '__main__':

    # SVM()
    # RFC()


    test_RFC()
    # rfc_feature_importance()

    # test_SVM()


    # csvdata  = pd.read_csv("csv/glcm/test/coal_with_light.csv")
    # data = csvdata.values
    # # print(data.shape)
    # data = np.split(data, [72], axis=1)
    #
    # X = data[0]
    # Y = data[1]
    # Y = Y.ravel()
    # print(X)
    # linear_svc = joblib.load("model/svm.m")
    #
    # print(linear_svc.predict(X))




