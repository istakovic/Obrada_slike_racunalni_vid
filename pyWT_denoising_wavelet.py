# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:29:14 2022

@author: IVAN
"""
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import cv2 as cv
import matplotlib.pyplot as plt

def gaussian_noise(img, mu, sigma):
  # generiraj nasumicne brojeve Gaussove distribucije
  noise = np.random.normal(mu, sigma, img.shape)
  return noise.astype(np.uint8)

image = plt.imread("slike/dog.jpeg")
original = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

#noisy na sebi ima gausov šum
noisy = cv.add(original,gaussian_noise(original, 0, 2))

plt.subplot(1,2,1)
plt.imshow(original, cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)

plt.imshow(noisy, cmap='gray')
plt.title("Gaussian noise")
plt.show()


titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
LL, (LH, HL, HH) = pywt.dwt2(noisy, 'db6')
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 15))
for i, a in enumerate([LL, LH, HL, HH]):
    ax[i].imshow(a, cmap=plt.cm.gray)
    ax[i].set_title(titles[i], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

fig.tight_layout()
plt.show()


titles = ['Noisy', 'Denoised']
denoised = pywt.idwt2((LL, (LH, HL, None)), 'db6')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
for i, a in enumerate([noisy, denoised]):
    ax[i].imshow(a, cmap=plt.cm.gray)
    ax[i].set_title(titles[i], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

fig.tight_layout()
plt.show()