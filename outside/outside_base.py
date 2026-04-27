#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abc import ABC, abstractmethod
import cv2

class OutsideBaseDetector(ABC):
    """
    抽象基类
    所有outside检测功能都可以继承这个类，实现统一的process方法
    """
    def __init__(self, **kwargs):
        """统一初始化入口，接收检测参数"""
        self.params = kwargs

    @abstractmethod
    def process(self, image: cv2.Mat):
        """
        统一的核心处理接口
        输入：原始BGR图像
        输出：固定格式元组 (flag, result_img, result_info, center, extra_data)
            flag: 检测是否成功/有效 0/1
            result_img: 绘制后的结果图像
            result_info: 检测结果的详细字典列表
            center: 检测目标的中心坐标 (x,y) 或 0
            extra_data: 额外返回数据（半径/宽度等）
        """
        pass

