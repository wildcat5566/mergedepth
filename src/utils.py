import numpy as np
import matplotlib.pyplot as plt
import cv2
        
def show_lidar(img, dots):
    if np.amax(img)==1.0:
        img = np.dot(img, 255).astype(np.uint8)
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(dots.shape[0]):
        for j in range(dots.shape[1]):
            if dots[i][j]:
                cv2.circle(canvas, (j,i), 2, (int(dots[i][j]),255,255),-1)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2RGB)
    plt.subplots(1,1, figsize = (18,5)) #13,3
    plt.title('Lidar data')
    plt.imshow(canvas)
