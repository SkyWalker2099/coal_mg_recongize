import cv2
import numpy as np
import os
# import tensorflow as tf
import ImageSegmentation
from matplotlib import pyplot as plt

def AlphsUpdate(img, a, b):
    img1 = np.uint8(np.clip((img * a + b), 0 ,255))
    return img1

def add_salt_pepper_noise(img):
    row,col,_ = img.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount*img.size*salt_vs_pepper)
    num_pepper = np.ceil(amount*img.size*(1.0 - salt_vs_pepper))

    coords = [np.random.random_integers(0, i-1, int(num_salt)) for i in img.shape]
    img[coords[0],coords[1],:] = 255
    coords = [np.random.random_integers(0, i - 1, int(num_salt)) for i in img.shape]
    img[coords[0],coords[1],:] = 0
    return img

def add_gaussian_noise(img):

    row,col,_ = img.shape
    var = 0.1

    gaussian = np.random.rand(row,col,1)*255
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian = gaussian*0.25

    print(gaussian)
    gaussian = np.array(gaussian, dtype=np.uint8)

    gaussian_img = cv2.addWeighted(img, 0.7, gaussian, 0.3, 0)
    # gaussian_img = np.array(gaussian_img, dtype=np.int32)
    print(gaussian_img.dtype)
    return gaussian_img

def Rotate90(img):
    center = (img.shape[1]/2,img.shape[0]/2)
    print(center)
    rot_mat = cv2.getRotationMatrix2D(center,90,1)
    img_rotated = cv2.warpAffine(img,rot_mat,(img.shape[1], img.shape[0]))
    return img_rotated


def allin():
    types1 = ["train", "test"]
    types2 = ["mg", "coal"]
    types3 = ["with_light", "without_light"]

    for t1 in types1:
        for t2 in types2:
            for t3 in types3:
                path = "CNN/data/" + t1 + "/" + t2 + "/" + t3 + "/"
                savepath = "CNN/data/Enhanced/" + t1 + "/" + t2 + "/" + t3 + "/"

                files = os.listdir(path)

                for file in files:
                    img = cv2.imread(path+file)

                    # cv2.imwrite(savepath+"ori_"+file, img)
                    #
                    # salt_img = add_salt_pepper_noise(img.copy())
                    # cv2.imwrite(savepath + "salt_"+file, salt_img)

                    # roated_img = Rotate90(img.copy())
                    # cv2.imwrite(savepath+"roated_"+file, roated_img)

                    if t3 == "with_light":
                        gaussian_img = add_gaussian_noise(img.copy())
                        cv2.imwrite(savepath+"gaussian_"+file, gaussian_img)




if __name__ == '__main__':
    allin()






