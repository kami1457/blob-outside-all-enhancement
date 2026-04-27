#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2

def angle_cos(p1, p2, p0):
    """计算三个点之间的角度余弦值"""
    d1, d2 = (p1 - p0).astype(np.float32), (p2 - p0).astype(np.float32)
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))

def calculate_line_distance(line1, line2):
    """计算两条平行直线之间的平均距离"""
    # 转换为直线方程：Ax + By + C = 0
    x1, y1, x2, y2 = line1
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = x2*y1 - x1*y2
    
    # 取四个端点计算平均距离
    points = [line2[:2], line2[2:]]
    distances = []
    for x, y in points:
        numerator = abs(A1*x + B1*y + C1)
        denominator = np.sqrt(A1**2 + B1**2)
        distances.append(numerator / denominator)
    
    return np.mean(distances)


# In[ ]:




