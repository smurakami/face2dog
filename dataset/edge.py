import os
import glob
import cv2
import numpy as np


def main():

    for i, file in enumerate(glob.glob('processed/*')):
        img = cv2.imread(file)


        blur = cv2.GaussianBlur(img, (5, 5), 4)

        edge = cv2.Canny(blur, 100, 200)
        edge = 255 - edge
        # h, w, d = img.shape
        # # print(h, w, d)
        # size = max(h, w)
        # scale = 256 / size

        # img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # h, w, d = img.shape
        # size = max(h, w)
        # print(h, w, d)

        # dst = np.zeros((size, size, d), dtype=img.dtype) + 255

        # if h > w:
        #     padding = (size - w) // 2
        #     dst[:, padding:padding+w] = img
        # elif h < w:
        #     padding = (size - h) // 2
        #     dst[padding:padding+h] = img

        cv2.imwrite('edge/dog_%04d.png' % i, edge)

    # print(i, file)
    # os.system('convert %s original_png/dog_%04d.png' % (file, i))


main()
