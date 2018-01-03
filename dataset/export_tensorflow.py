import numpy as np

import os
import glob
import cv2
import numpy as np


def main():

    # image_list = []
    # edge_list = []

    for i, file in enumerate(sorted(glob.glob('selected/*'))):
        print(file)
        base = os.path.basename(file)
        # image_file = 'processed/' + base

        img = cv2.imread('processed/%s' % base)[:, :, ::-1]
        cv2.imwrite('tensorflow/image/%s' % base, img)

        # os.system('cp processed/%s tensorflow/image/%s' % (base, base))
        os.system('cp edge/%s tensorflow/edge/%s' % (base, base))

        # image = cv2.imread(image_file)
        # image_list.append(image)
        # edge_file = 'edge/' + base

        # edge = cv2.imread(edge_file)
        # edge_list.append(edge)

        # # print(base)
        # print(image.shape)
        # print(edge.shape)


    # image = np.array(image_list)
    # edge = np.array(edge_list)

    # print(image.shape)
    # print(edge.shape)

    # np.save('../data/image.npy', image)
    # np.save('../data/edge.npy', edge)

        # img = cv2.imread(file)
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
        # else:
        #     dst = img

        # cv2.imwrite('processed/dog_%04d.png' % i, dst)

    # print(i, file)
    # os.system('convert %s original_png/dog_%04d.png' % (file, i))


main()



