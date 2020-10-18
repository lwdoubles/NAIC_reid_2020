import os
import numpy as np
import cv2

# 初赛A训练数据分布
# R_mean: 96.644014, G_mean: 96.940007, B_mean: 101.886814
# R_std: 64.393160, G_std: 63.946692, B_std: 55.728104
# 初赛A测试数据分布
# R_mean: 77.135664, G_mean: 83.419097, B_mean: 82.425086
# R_std: 60.537581, G_std: 51.536042, B_std: 53.111341

# files_dir = '../contest/train/images/'
files_dir = '../contest/image_A/query/'
files = os.listdir(files_dir)

R = 0.
G = 0.
B = 0.
R_2 = 0.
G_2 = 0.
B_2 = 0.
N = 0

for file in files:
    img = cv2.imread(files_dir+file, 1)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        h, w, c = img.shape
        N += h*w

        R_t = img[:, :, 0]
        R += np.sum(R_t)
        R_2 += np.sum(np.power(R_t, 2.0))

        G_t = img[:, :, 1]
        G += np.sum(G_t)
        G_2 += np.sum(np.power(R_t, 2.0))

        B_t = img[:, :, 2]
        B += np.sum(B_t)
        B_2 += np.sum(np.power(R_t, 2.0))

R_mean = R/N
G_mean = G/N
B_mean = B/N

R_std = np.sqrt(R_2/N - R_mean*R_mean)
G_std = np.sqrt(G_2/N - G_mean*G_mean)
B_std = np.sqrt(B_2/N - B_mean*B_mean)

print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))