#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
from outside_base import OutsideBaseDetector
from outside_utils import angle_cos, calculate_line_distance


# 1. 最大椭圆检测
class MaxEllipseDetector(OutsideBaseDetector):
    def process(self, image: cv2.Mat):
        # 仅适配统一接口
        if image is None:
            return 0, image, [], (0, 0), None

        canny_threshold = self.params.get('canny_threshold', 20)
        min_contour_area = self.params.get('min_contour_area', 250)
        aspect_ratio_tol = self.params.get('aspect_ratio_tol', 0.2)
        start_area = self.params.get('start_area', 250)

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 高斯滤波降噪
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # Canny边缘检测
        edges = cv2.Canny(blurred, canny_threshold // 2, canny_threshold)
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        result_img = image.copy()
        max_ellipse = None
        max_area = start_area
        contour_max = None

        for contour in contours:
            # 过滤小面积轮廓
            if cv2.contourArea(contour) < min_contour_area:
                continue
            # 至少需要5个点来拟合椭圆
            if contour.shape[0] < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(contour)
                # 计算椭圆参数
                (x, y), (a, b), angle = ellipse
                current_area = np.pi * a * b
                # 过滤不合理的椭圆（纵横比容差）
                if abs(a - b) / max(a, b) > aspect_ratio_tol:
                    continue
                # 更新最大椭圆
                if current_area > max_area:
                    max_area = current_area
                    max_ellipse = ellipse
                    contour_max = contour
                # 绘制所有有效椭圆（绿色）
                cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
            except:
                continue

        # 绘制最大椭圆（红色）
        flag = 0
        center = (0, 0)
        if max_ellipse is not None:
            flag = 1
            # 绘制椭圆
            cv2.ellipse(result_img, max_ellipse, (0, 0, 255), 4)
            # 绘制中心点
            center = tuple(map(int, max_ellipse[0]))
            cv2.circle(result_img, center, 5, (255, 255, 0), 8)
            # 添加文字标注
            cv2.putText(result_img, "Max Ellipse", (center[0] - 50, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # 统一接口返回格式
        return flag, result_img, [{"max_ellipse": max_ellipse}], center, contour_max

# 2. 椭圆检测（原detect_ellipses函数）
class EllipseDetector(OutsideBaseDetector):
    def process(self, image: cv2.Mat):
        # 仅适配统一接口
        flag = 0
        result_img = image.copy()
        ellipse_info = []
        center = (0, 0)
        r_max = 0

        if image is None:
            return flag, result_img, ellipse_info, center, r_max

        canny_threshold = self.params.get('canny_threshold', 20)
        min_contour_area = self.params.get('min_contour_area', 250)
        bais_area = self.params.get('bais_area', 0.2)
        start_area = self.params.get('start_area', 250)
        ratio_limit = self.params.get('ratio_limit', 0.5)

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 高斯滤波降噪
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        # Canny边缘检测
        edges = cv2.Canny(blurred, canny_threshold//2, canny_threshold)
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 过滤小面积轮廓
            if cv2.contourArea(contour) < min_contour_area:
                continue
            # 至少需要5个点来拟合椭圆
            if contour.shape[0] < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(contour)
                # 计算椭圆参数
                (x, y), (a, b), angle = ellipse
                current_area = np.pi * a * b
                r = math.sqrt(a * b)

                if a == 0 or b == 0:
                    continue
                if min(a/b, b/a) < ratio_limit:
                    continue

                ellipse_info.append({
                    "center": (x, y),
                    "axes": (a, b),
                    "angle": angle,
                    "r": r,
                    "area": current_area,
                    "ellipse": ellipse
                })
                # 绘制所有有效椭圆（绿色）
                cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
            except:
                continue

        # 排序找第二大椭圆
        sorted_ellipse_info = sorted(ellipse_info, key=lambda info: info["area"], reverse=True)
        radius_diff_threshold = 0.1
        if len(sorted_ellipse_info) == 0:
            return flag, result_img, ellipse_info, center, r_max

        largest_ellipse = sorted_ellipse_info[0]
        second_largest_ellipse = None

        for idx in range(1, len(sorted_ellipse_info)):
            current_ellipse = sorted_ellipse_info[idx]
            max_radius = np.sqrt(largest_ellipse["area"] / np.pi)
            current_radius = np.sqrt(current_ellipse["area"] / np.pi)
            radius_difference = abs(max_radius - current_radius) / max_radius

            if radius_difference <= radius_diff_threshold:
                continue
            else:
                second_largest_ellipse = current_ellipse
                break

        if second_largest_ellipse is not None:
            flag = 1
            cv2.ellipse(result_img, second_largest_ellipse["ellipse"], (0, 0, 255), 4)
            center = tuple(map(int, second_largest_ellipse["center"]))
            r_max = int(second_largest_ellipse["r"])
            cv2.circle(result_img, center, 5, (255, 255, 0), 8)
            cv2.putText(result_img, "True Second Largest Ellipse", (center[0] - 50, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 统一接口返回格式
        return flag, result_img, ellipse_info, center, r_max


# 3. 梯形检测
class TrapezoidDetector(OutsideBaseDetector):
    def process(self, image: cv2.Mat):
        # 原代码逻辑原封不动保留，仅适配统一接口
        flag = 0
        result_img = image.copy()
        trapezoid_info = []
        center_max = (0, 0)
        width_max = 0

        if image is None:
            print("Error: Image not loaded")
            return flag, result_img, trapezoid_info, center_max, width_max

        max_area = self.params.get('max_area', 10)
        cos_max = self.params.get('cos_max', 0.85)
        min_contour_area = self.params.get('min_contour_area', 100)

        # 1. 转灰度图
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        # 2. 中值滤波
        blurred = cv2.medianBlur(gray, 5)
        # 3. Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        # 4. 膨胀操作
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        # 5. 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        trapezoids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue
            # 多边形逼近
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 检测4边形且为凸多边形
            if len(approx) == 4 and cv2.isContourConvex(approx):
                cosines = []
                for i in range(4):
                    p0, p1, p2 = approx[i][0], approx[(i+1)%4][0], approx[(i+2)%4][0]
                    cos = angle_cos(p1, p2, p0)
                    cosines.append(cos)
                max_cos = max(np.abs(cosines))
                if max_cos < cos_max:
                    approx = approx.reshape(-1, 2)
                    trapezoids.append(approx)
                    area = cv2.contourArea(approx)
                    if area > max_area:
                        flag = 1
                        max_area = area
                        max_trap = cnt.reshape(-1, 2)
                        center_max = (int(np.mean(max_trap[:, 0])), int(np.mean(max_trap[:, 1])))
                        rect_max = cv2.minAreaRect(max_trap)
                        (w_max, h_max) = rect_max[1]
                        width_max = int(min(w_max, h_max))

        for approx in trapezoids:
            approx = approx.reshape(-1, 2)
            center = (int(np.mean(approx[:, 0])), int(np.mean(approx[:, 1])))
            rect = cv2.minAreaRect(approx)
            (w, h) = rect[1]
            trapezoid_info.append({
                "contour": approx, 
                "center": center,
                "width": int(min(w, h))
            })

        for trap in trapezoids:
            cv2.polylines(result_img, [trap.reshape(-1, 1, 2).astype(int)], True, (0, 255, 0), 3)
            center_x = int(np.mean(trap[:, 0]))
            center_y = int(np.mean(trap[:, 1]))
            cv2.putText(result_img, "Trapezoid", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 统一接口返回格式
        return flag, result_img, trapezoid_info, center_max, width_max

# 4. 三角形检测（原detect_triangle函数）
class TriangleDetector(OutsideBaseDetector):
    def process(self, image: cv2.Mat):
        # 仅适配统一接口
        flag = 0
        result_img = image.copy()
        triangles_info = []
        center_max = (0, 0)
        radius_max = 0

        if image is None:
            print("Error: Image not loaded")
            return flag, result_img, triangles_info, center_max, radius_max

        max_area = self.params.get('max_area', 10)
        cos_max = self.params.get('cos_max', 0.85)
        min_contour_area = self.params.get('min_contour_area', 100)

        # 1. 转灰度图
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        # 2. 中值滤波
        blurred = cv2.medianBlur(gray, 9)
        # 3. Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        # 4. 膨胀操作
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        # 5. 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        triangles = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue
            # 多边形逼近
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 检测3边形且为凸多边形
            if len(approx) == 3 and cv2.isContourConvex(approx):
                cosines = []
                for i in range(3):
                    p0, p1, p2 = approx[i][0], approx[(i+1)%3][0], approx[(i+2)%3][0]
                    cos = angle_cos(p1, p2, p0)
                    cosines.append(cos)
                max_cos = max(np.abs(cosines))
                if max_cos < cos_max:
                    approx = approx.reshape(-1, 2)
                    triangles.append(approx)
                    area = cv2.contourArea(approx)
                    if area > max_area:
                        flag = 1
                        max_area = area
                        max_trap = cnt.reshape(-1, 2)
                        (center_max, radius) = cv2.minEnclosingCircle(max_trap)
                        radius_max = int(radius)
                        center_max = (int(center_max[0]), int(center_max[1]))

        for approx in triangles:
            approx = approx.reshape(-1, 2)
            (center, radius) = cv2.minEnclosingCircle(approx)
            radius = int(radius)
            triangles_info.append({
                "contour": approx, 
                "center": center, 
                "radius": radius
            })

        for trap in triangles:
            cv2.polylines(result_img, [trap.reshape(-1, 1, 2).astype(int)], True, (0, 255, 0), 3)
            center_x = int(np.mean(trap[:, 0]))
            center_y = int(np.mean(trap[:, 1]))
            cv2.putText(result_img, "Triangle", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 统一接口返回格式
        return flag, result_img, triangles_info, center_max, radius_max

# 5. 直线检测（原find_longest_straight_line函数）
class LineDetector(OutsideBaseDetector):
    def process(self, image: cv2.Mat):
        # 仅适配统一接口
        flag = 0
        result_img = image.copy()
        pole_groups = []
        center = 0

        if image is None:
            print("Error: Image not loaded")
            return flag, result_img, pole_groups, center, 0

        vertical_angle_threshold = self.params.get('vertical_angle_threshold', 10)
        min_line_length = self.params.get('min_line_length', 100)
        max_gap = self.params.get('max_gap', 20)

        # 1. 图像预处理
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,5)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        # 2. 霍夫变换检测线条
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=min_line_length, 
                                maxLineGap=max_gap)

        if lines is None:
            return flag, result_img, pole_groups, center, 0

        # 角度分组逻辑
        angle_groups = {}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            cv2.line(result_img, (x1,y1), (x2,y2), (0,0,255), 2)
            theta = np.arctan2(dy, dx) * 180 / np.pi
            if theta < 0:
                theta += 180
            if theta > 180:
                theta -= 180

            # 过滤非垂直线条
            vertical_angle = np.abs(np.abs(theta) - 90)
            if vertical_angle > vertical_angle_threshold:
                continue
            length = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 180

            angle_key = round(angle / 3) * 3
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            quality = length * 0.8 + cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)[-1] * 0.2

            if angle_key not in angle_groups:
                angle_groups[angle_key] = {'lines': [], 'total_quality': 0}

            group = angle_groups[angle_key]
            if len(group['lines']) < 2:
                group['lines'].append((quality, line[0], angle))
                group['total_quality'] += quality
                group['lines'].sort(reverse=True, key=lambda x: x[0])
            else:
                if quality > group['lines'][-1][0]:
                    group['total_quality'] -= group['lines'][-1][0]
                    group['lines'].pop()
                    group['lines'].append((quality, line[0], angle))
                    group['total_quality'] += quality
                    group['lines'].sort(reverse=True, key=lambda x: x[0])

        # 平行线组处理
        best_groups = sorted(angle_groups.values(), key=lambda x: -x['total_quality'])[:20]
        for group in best_groups:
            if len(group['lines']) >= 2:
                line1 = group['lines'][0][1]
                line2 = group['lines'][1][1]
                avg_distance = calculate_line_distance(line1, line2)
                if 5 < avg_distance < 70:
                    flag = 1
                    x_coords = [line[1][0] for line in group['lines']]
                    x2_coords = [line[1][2] for line in group['lines']]
                    avg_x = np.mean([(x1 + x2) / 2 for x1, x2 in zip(x_coords, x2_coords)])
                    pole_groups.append({
                        'angle': group['lines'][0][2],
                        'lines': [line[1] for line in group['lines']],
                        'avg_distance': avg_distance,
                        'center': [int(avg_x)]
                    })

        # 绘制逻辑
        color_anlysis = (0, 0, 255)
        for pole in pole_groups:
            count = 0
            for line in pole['lines']:
                x1, y1, x2, y2 = line
                if count%2 == 0:
                    color_anlysis = (0, 0, 255)
                else:
                    color_anlysis = (255, 0, 0)
                count += 1
                cv2.line(result_img, (x1,y1), (x2,y2), color_anlysis, 2)
            for center_x in pole['center']:
                cv2.line(result_img, (center_x, 0), (center_x, 470), (0, 255, 0), 2)

        if pole_groups:
            center = pole_groups[0]["center"]

        # 统一接口返回格式
        return flag, result_img, pole_groups, center, 0


# In[ ]:




