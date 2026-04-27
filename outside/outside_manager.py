#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from outside_detectors import (
    MaxEllipseDetector,
    EllipseDetector,
    TrapezoidDetector,
    TriangleDetector,
    LineDetector
)

class OutsideManager:
    """
    统一管理类
    对外唯一入口，封装所有outside检测功能，统一调用方式
    """
    def __init__(self):
        # 预注册所有检测器
        self._detectors = {
            "max_ellipse": MaxEllipseDetector,
            "ellipse": EllipseDetector,
            "trapezoid": TrapezoidDetector,
            "triangle": TriangleDetector,
            "line": LineDetector
        }

    def detect(self, detect_type: str, image: cv2.Mat, **kwargs):
        """
        统一检测入口，radar5的统一方法设计
        :param detect_type: 检测类型，可选：max_ellipse/ellipse/trapezoid/triangle/line
        :param image: 输入BGR图像
        :param kwargs: 检测参数，和原函数参数完全一致
        :return: 统一格式结果 (flag, result_img, result_info, center, extra_data)
        """
        if detect_type not in self._detectors:
            print(f"错误：不支持的检测类型 {detect_type}，支持的类型：{list(self._detectors.keys())}")
            return 0, image, [], (0, 0), None

        # 实例化检测器，传入参数
        detector = self._detectors[detect_type](**kwargs)
        # 调用统一process接口，实现多态
        return detector.process(image)

