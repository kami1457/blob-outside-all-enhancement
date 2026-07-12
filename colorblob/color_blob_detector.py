"""色块检测器。"""

import cv2
import numpy as np


class ColorBlobDetector:
    def __init__(self, color_bounds, min_area=800):
        self.color_bounds = color_bounds
        self.min_area = min_area

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

    def detect(self, frame):
        # 降低图像噪点
        blurred = cv2.GaussianBlur(
            frame,
            (5, 5),
            0,
        )

        # RGB888 从 Picamera2 取出后，可以按照 OpenCV BGR 图像处理
        hsv_image = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2HSV,
        )

        mask = np.zeros(
            hsv_image.shape[:2],
            dtype=np.uint8,
        )

        # 合并该颜色的所有 HSV 范围
        for lower, upper in self.color_bounds:
            current_mask = cv2.inRange(
                hsv_image,
                lower,
                upper,
            )

            cv2.bitwise_or(
                mask,
                current_mask,
                dst=mask,
            )

        # 去除小噪点
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )

        # 填补物体内部的小孔洞
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=2,
        )

        # 查找外部轮廓
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 根据面积过滤轮廓
        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.min_area
        ]

        # 按面积从大到小排列
        valid_contours.sort(
            key=cv2.contourArea,
            reverse=True,
        )

        return valid_contours, mask
