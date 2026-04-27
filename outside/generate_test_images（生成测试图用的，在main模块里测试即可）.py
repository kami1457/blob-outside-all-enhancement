import cv2
import numpy as np

# 统一设置：黑底白图形，高对比度
BG_COLOR = (0, 0, 0)  # 黑色背景
FG_COLOR = (255, 255, 255)  # 白色图形，高对比度
IMG_SIZE = (480, 640)  # 图片尺寸：高480，宽640，适配你代码的直线检测高度

# ===================== 1. 生成【最大椭圆检测】测试图 =====================
img_max_ellipse = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), BG_COLOR, dtype=np.uint8)
# 画一个正圆（椭圆的特殊形式），半径120像素，面积足够大，不会被过滤
cv2.circle(img_max_ellipse, (320, 240), 120, FG_COLOR, -1)
cv2.imwrite("max_ellipse_test.png", img_max_ellipse)
print("已生成：max_ellipse_test.png（最大椭圆检测用）")

# ===================== 2. 生成【椭圆检测】测试图 =====================
img_ellipse = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), BG_COLOR, dtype=np.uint8)
# 画两个大小差30%的正圆
cv2.circle(img_ellipse, (180, 240), 120, FG_COLOR, -1)  # 大圆
cv2.circle(img_ellipse, (460, 240), 80, FG_COLOR, -1)   # 小圆（面积差30%以上）
cv2.imwrite("ellipse_test.png", img_ellipse)
print("已生成：ellipse_test.png（椭圆检测用）")

# ===================== 3. 生成【梯形检测】测试图 =====================
img_trapezoid = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), BG_COLOR, dtype=np.uint8)
# 画一个凸梯形
pts = np.array([[200, 120], [440, 120], [480, 360], [160, 360]], np.int32)
cv2.fillPoly(img_trapezoid, [pts], FG_COLOR)
cv2.imwrite("trapezoid_test.png", img_trapezoid)
print("已生成：trapezoid_test.png（梯形检测用）")

# ===================== 4. 生成【三角形检测】测试图 =====================
img_triangle = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), BG_COLOR, dtype=np.uint8)
# 画一个等腰凸三角形
pts = np.array([[320, 100], [160, 380], [480, 380]], np.int32)
cv2.fillPoly(img_triangle, [pts], FG_COLOR)
cv2.imwrite("triangle_test.png", img_triangle)
print("已生成：triangle_test.png（三角形检测用）")

# ===================== 5. 生成【直线检测】测试图 =====================
img_line = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), BG_COLOR, dtype=np.uint8)
# 画两条平行竖线，长度200像素，间距30像素
cv2.line(img_line, (295, 140), (295, 340), FG_COLOR, 2)
cv2.line(img_line, (345, 140), (345, 340), FG_COLOR, 2)
cv2.imwrite("line_test.png", img_line)
print("已生成：line_test.png（直线检测用）")

print("\n所有测试图生成完成！已保存到当前项目文件夹")