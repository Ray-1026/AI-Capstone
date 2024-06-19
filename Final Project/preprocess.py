import numpy as np
import os
import cv2
from PIL import Image


def preprocess(path):
    for i in os.listdir(path):
        fp = os.path.join(path, i)
        print(fp)

        img = Image.open(fp)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.merge([img, img, img])

        cv2.imwrite(fp, img)


# def preprocess():
#     path = "our_dataset"
#     f = np.loadtxt("DAVIS/ImageSets/480p/val.txt", dtype=str)
#     print(f.shape)

#     for i in f:
#         print(i)
#         name, img_name1 = i[0].split("/")[3:5]
#         img_name2 = i[1].split("/")[4]
#         newpath = os.path.join(path, name)

#         # create folder
#         if not os.path.exists(os.path.join(path, name)):
#             os.makedirs(os.path.join(path, name))
#         if not os.path.exists(os.path.join(newpath, name)):
#             os.makedirs(os.path.join(newpath, name))
#         if not os.path.exists(os.path.join(newpath, f"{name}_masks")):
#             os.makedirs(os.path.join(newpath, f"{name}_masks"))

#         # copy files
#         os.system(f"cp DAVIS{i[0]} {newpath}/{name}/{img_name1}")
#         os.system(f"cp DAVIS{i[1]} {newpath}/{name}_masks/{img_name2}")

#         # RGB to Gray
#         img = Image.open(f"{newpath}/{name}/{img_name1}")
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img = cv2.merge([img, img, img])

#         cv2.imwrite(f"{newpath}/{name}/{img_name1}", img)


if __name__ == "__main__":
    preprocess()
