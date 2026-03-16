import cv2
import numpy as np
import os


class HSVColorRegistry:
    def __init__(self):
        # 预定义各种颜色的 HSV 阈值区间
        # OpenCV 中 H: 0-179, S: 0-255, V: 0-255
        self.colors = {
            'red': [
                ([0, 192, 165], [1, 255, 255]),
                ([179, 192, 165], [179, 255, 255])
            ],
            'orange': [
                ([7, 100, 100], [19, 255, 255]),
                ([3, 100, 100], [2, 255, 255])
            ],
            'yellow': [
                ([25, 100, 100], [32, 255, 255]),
                ([13, 100, 100], [11, 255, 255])
            ],
            'green': [
                ([42, 80, 80], [83, 255, 255]),
                ([3, 80, 80], [2, 255, 255])
            ],
            'cyan': [
                ([70, 80, 80], [88, 255, 255]),
                ([2, 80, 80], [1, 255, 255])
            ],
            'blue': [
                ([90, 80, 80], [123, 255, 255]),
                ([2, 80, 80], [1, 255, 255])
            ],
            'purple': [
                ([129, 80, 80], [159, 255, 255]),
                ([4, 80, 80], [3, 255, 255])
            ],
            'white': [
                ([0, 0, 243], [30, 40, 255])
            ],
            'black': [
                ([0, 0, 0], [179, 255, 58])
            ],
            'gray': [
                ([0, 0, 50], [83, 32, 104])
            ]
        }

    def get_color_bounds(self, color_name):
        """获取指定颜色的 HSV 阈值"""
        color_name = color_name.lower()
        if color_name in self.colors:
            return self.colors[color_name]
        else:
            print(f"警告: 不支持的颜色 '{color_name}'。支持的颜色有: {self.available_colors()}")
            return None

    def available_colors(self):
        """返回所有支持的颜色列表"""
        return list(self.colors.keys())

    def add_custom_color(self, name, bounds_list):
        """允许在代码运行时动态添加自定义颜色"""
        self.colors[name.lower()] = bounds_list


# ==========================================
# 图像处理模块
# ==========================================
class Preprocessor:
    def __init__(self, resize_factor=0.5):
        self.resize_factor = resize_factor

    def process(self, image):
        if self.resize_factor != 1:
            width = int(image.shape[1] * self.resize_factor)
            height = int(image.shape[0] * self.resize_factor)
            image = cv2.resize(image, (width, height))
        return image


class ColorFilter:
    def __init__(self, color_bounds):
        self.color_bounds = color_bounds

    def apply(self, image):
        # 使用 GaussianBlur 先进行平滑处理，能极大改善噪点和边缘毛刺，让识别更稳定
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_bounds:
            current_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, current_mask)
        return mask


class MorphologicalOperation:
    def __init__(self, kernel_size=(5, 5)):
        # 稍微增大内核，有助于把破碎的同色块连接起来
        self.kernel = np.ones(kernel_size, np.uint8)

    def apply(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        # 增加一次轻微膨胀，使得轮廓更平滑完整
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        return mask


class ContourExtractor:
    def __init__(self, min_area=100):
        self.min_area = min_area

    def extract(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        return valid_contours


class ColorBlobDetector:
    def __init__(self, color_bounds, resize_factor=0.5, min_area=100):
        self.preprocessor = Preprocessor(resize_factor)
        self.color_filter = ColorFilter(color_bounds)
        self.morph_op = MorphologicalOperation(kernel_size=(7, 7))
        self.contour_extractor = ContourExtractor(min_area)

    def detect(self, image):
        processed_image = self.preprocessor.process(image)
        mask = self.color_filter.apply(processed_image)
        clean_mask = self.morph_op.apply(mask)
        contours = self.contour_extractor.extract(clean_mask)
        return processed_image, contours, mask


# ==========================================
# 图像绘制与标注模块 (新增类)
# ==========================================
class ResultAnnotator:
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def annotate(self, image, contours, target_color):
        # 复制一份图像以防修改原始处理图像
        annotated_image = image.copy()

        for contour in contours:
            # ====================================
            # 优化 1: 多边形逼近 (Polygonal Approximation)
            # ====================================
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 画出紧贴色块边缘的真实多边形边界框 (红色粗线)
            cv2.drawContours(annotated_image, [approx], 0, (0, 0, 255), 2)

            # 仅获取标准的 (x, y) 坐标用于定位文字标签
            x, y, w, h = cv2.boundingRect(contour)

            # ====================================
            #  计算并绘制色块的质量中心点 (重心)
            # ====================================
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # 在中心点画一个黄色的实心准星圆点
                cv2.circle(annotated_image, (cx, cy), 4, (0, 255, 255), -1)

            # ====================================
            #  带有抗干扰背景色的高亮标签显示
            # ====================================
            label = f"{target_color.upper()}"
            (text_w, text_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)

            # 确保文字背景框不会超出图像上边界导致崩溃
            bg_y = max(0, y - text_h - 10)

            # 绘制蓝色文字背景块
            cv2.rectangle(annotated_image, (x, bg_y), (x + text_w, bg_y + text_h + 10), (255, 0, 0), -1)
            # 绘制白色文字
            cv2.putText(annotated_image, label, (x, bg_y + text_h + 5), self.font, self.font_scale, (255, 255, 255),
                        self.thickness)

        return annotated_image


# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    # 1. 实例化各个功能模块的类
    color_registry = HSVColorRegistry()
    annotator = ResultAnnotator()

    # 2. 设置检测参数
    target_color = 'blue'
    image_path = r"D:\Users\pengy\Downloads\chatgpt.png"

    # 3. 检查和加载数据
    color_bounds = color_registry.get_color_bounds(target_color)
    if color_bounds is None:
        exit()

    if not os.path.isfile(image_path):
        print(f"无法找到图像文件: {image_path}")
        exit()

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        exit()

    # 4. 初始化检测器并执行检测核心逻辑
    detector = ColorBlobDetector(color_bounds, resize_factor=0.5, min_area=800)
    processed_image, contours, mask = detector.detect(image)

    # 5. 调用标注类，在图像上画出所有的轮廓、中心点和文字框
    final_image = annotator.annotate(processed_image, contours, target_color)

    # 6. 显示最终结果
    cv2.imshow(f'Detected {target_color.upper()} Blobs', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()