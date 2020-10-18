import os 
import numpy as np
import cv2

files_dir = '../contest/train/images/'
files = os.listdir(files_dir)

R = 0.
G = 0.
B = 0.
R_2 = 0.
G_2 = 0.
B_2 = 0.
N = 0

anomaly = []


with open('./anomaly.txt', 'w') as ano_file:
    for file in files:
        img = cv2.imread(files_dir+file, 1)
        if img is None:
            ano_file.write(files_dir+file)
            ano_file.write('\n')