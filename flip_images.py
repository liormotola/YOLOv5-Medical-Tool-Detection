import os
from os import listdir
from os.path import isfile, join
import torch
import glob

import cv2
from tqdm import tqdm

train_files = glob.glob("datasets/train/images/*") + glob.glob("datasets/valid/images/*") + glob.glob(
    "datasets/test/images/*")
train_labels = glob.glob("datasets/train/labels/*") + glob.glob("datasets/valid/labels/*") + glob.glob(
    "datasets/test/labels/*")


fix_label = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7}
switch = {i: i + 1 if i % 2 == 0 else i - 1 for i in range(8)}

for image in tqdm(train_files):
    try:
        originalImage = cv2.imread(image)
        # cv2.imshow("bla",originalImage)
        flipHorizontal = cv2.flip(originalImage, 1)
        # cv2.imshow("bla2",flipHorizontal)
        cv2.imwrite(image.replace(".jpg", "_flipped.jpg"), flipHorizontal)
        with open(image.replace(".jpg", ".txt").replace("images", 'labels')) as f:
            labels = f.readlines()

        new_labels = [line.split(" ") for line in labels]
        to_replace = []
        to_fix = []
        for line in new_labels:
            shape = originalImage.shape
            label, x, y, w, h = line
            label = str(fix_label[int(label)])
            new_label = str(switch[int(label)])
            new_x = float(x.strip()) * shape[0]
            y = str(float(y.strip()))
            w = str(float(w.strip()))
            h = str(float(h.strip())) + "\n"
            new_x = str(abs(shape[0] - new_x) / shape[0])
            to_replace.append(" ".join([new_label, new_x, y, w, h]))
            to_fix.append(" ".join([label, x, y, w, h]))
            with open(image.replace(".jpg", "_flipped.txt").replace("images", 'labels'), 'w') as f:
                for line in to_replace:
                    f.write(line)
            with open(image.replace(".jpg", ".txt").replace("images", 'labels'), 'w') as f:
                for line in to_fix:
                    f.write(line)
    except Exception as e:
        print(e)
        continue
