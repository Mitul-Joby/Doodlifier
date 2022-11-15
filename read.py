
import cv2
import matplotlib.pyplot as plt
import numpy as np

custom = cv2.imread('canvas.png', cv2.IMREAD_GRAYSCALE)
custom = cv2.resize(custom, (28, 28))
custom = np.array(custom)
custom = custom.reshape(28, 28, 1).astype('float32')
custom /= 255.0
custom = 1 - custom

plt.imshow(custom.squeeze()) 
plt.show()