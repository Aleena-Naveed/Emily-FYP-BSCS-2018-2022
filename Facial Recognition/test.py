import cv2
import numpy as np

path = r'E:\Emily\Facial Expresion Recognation\images\train\angry\0.jpg'
img = cv2.imread(path)
print(np.shape(img))
