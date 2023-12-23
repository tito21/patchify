from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

def build_gabor_filter(size, sigma, theta, lam, gamma):
    kernel = cv2.getGaborKernel((size, size), sigma, theta, lam, gamma)
    return kernel

def save_image(img, name):
    img = img[:-1, :-1]
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(name, img)

size = 224

sigmas = np.linspace(15, 75.0, 50)
thetas = np.linspace(0, np.pi, 10)
lams = np.linspace(10, 100, 25)
gammas = np.linspace(0.1, 1.0, 25)

num_filters = len(sigmas) * len(thetas) * len(lams) * len(gammas)
print(num_filters)

for index, (sigma, theta, lam, gamma) in tqdm(enumerate(product(sigmas, thetas, lams, gammas)), total=num_filters):
    filter = build_gabor_filter(size, sigma, theta, lam, gamma)
    # print(filter.shape, filter.min(), filter.max())
    # plt.imshow(filter, cmap="gray")
    # plt.show()
    save_image(filter, f"gabor-filters/{index:07d}_{sigma:0.3f}_{theta:0.3f}_{lam:0.3f}_{gamma:0.3f}.png")
    # break

