# # import ImageSegmentation1
from ImageSegmentation1 import *
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image,ImageTk
import numpy as np
import cv2
import os


def choosepic():
    path_ = askopenfilename()
    print(path_)
    path.set(path_)

    image_open = Image.open(path_)
    img = ImageTk.PhotoImage(image_open)
    l1.config(image = img)
    l1.image = img

def taret_detect():
    p = path.get()
    # img = cv2.imread(p)
    # print(img.shape)
    # print(img)
    img_with_labels,c,boxs = pic_cmr(p)
    pil_img = Image.fromarray(img_with_labels.astype('uint8')).convert('RGB')
    img = ImageTk.PhotoImage(pil_img)
    l1.config(image=img)
    l1.image = img


root = Tk()
path = StringVar()
Button(root, text="选择图片", command=choosepic).pack()
Button(root, text="目标检测", command=taret_detect).pack()
e1 = Entry(root, state="readonly", text=path)
e1.pack()
l1 = Label(root)
l1.pack()

root.mainloop()





# if __name__ == '__main__':
    # path = "mult/"
    # files = os.listdir(path)
    # for file in files:
    #     img = cv2.imread(path+file)
    #     if(img.shape[0]>900):
    #         h = 900
    #         w = int(900*(img.shape[1]/img.shape[0]))
    #         img = cv2.resize(img,dsize=(w,h))
    #         cv2.imwrite(file,img)

