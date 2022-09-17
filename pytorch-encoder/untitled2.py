# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 21:04:46 2021

@author: Dell
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_org = cv2.imread('C:/Users/Dell/OneDrive/Desktop/IMG_20190814_104622.jpg')
rows, cols, ch = img_org.shape
color = [100, 0, 0]
img = cv2.copyMakeBorder(img_org.copy(),100,100,100,100,cv2.BORDER_CONSTANT,value=color)

(cX, cY) = (cols// 2, rows // 2)
M = cv2.getRotationMatrix2D((cX, cY), 11, 1.0)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imwrite('C:/Users/Dell/OneDrive/Desktop/edit.jpg',dst)

im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(im_rgb)
plt.title('Input')
plt.savefig
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()

# Displaying the image
# while(1):
# 	
# 	cv2.imshow('image', img)
# 	if cv2.waitKey(20) & 0xFF == 27:
# 		break
# 		
# cv2.destroyAllWindows()
