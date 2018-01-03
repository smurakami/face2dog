import os
import glob
import cv2
import numpy as np


def main():

    for i, file in enumerate(sorted(glob.glob('original/*'))):
        img = cv2.imread(file)[:, :, ::-1]
        h, w, d = img.shape
        # print(h, w, d)
        size = max(h, w)
        scale = 256 / size
        if w > h:
            img = cv2.resize(img, (256, int(h * scale)), interpolation=cv2.INTER_AREA)
        elif w < h:
            img = cv2.resize(img, (int(w * scale), 256), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        h, w, d = img.shape
        size = max(h, w)
        print(h, w, d)

        dst = np.zeros((size, size, d), dtype=img.dtype) + 255

        if h > w:
            padding = (size - w) // 2
            dst[:, padding:padding+w] = img
        elif h < w:
            padding = (size - h) // 2
            dst[padding:padding+h] = img
        else:
            dst = img

        cv2.imwrite('processed/dog_%04d.png' % i, dst)

    # print(i, file)
    # os.system('convert %s original_png/dog_%04d.png' % (file, i))


main()
