#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
from outside_manager import OutsideManager

def main():
    # 1. 初始化统一管理器（radar5的RadarManager初始化）
    outside_manager = OutsideManager()

    # 2. 配置参数
    detect_type = "max_ellipse"  # 直接选择测试的模块，即：max_ellipse/ellipse/trapezoid/triangle/line
    image_path = r"max_ellipse_test.png"  # 替换为你的测试图路径（命名随意，放在同一个文件夹即可）

    # 3. 校验图片路径
    if not os.path.isfile(image_path):
        print(f"无法找到图像文件: {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        return

    # 4. 调用统一检测接口
    # 可传入原函数的所有参数，比如canny_threshold=30, min_area=500等
    flag, result_img, result_info, center, extra_data = outside_manager.detect(
        detect_type=detect_type,
        image=image,
        canny_threshold=20,
        min_contour_area=250
    )

    # 5. 输出结果日志（radar5的日志输出）
    print(f"检测结果：flag={flag}, 中心坐标={center}, 额外数据={extra_data}")
    print(f"检测到的目标数量：{len(result_info)}")

    # 6. 显示结果
    cv2.imshow(f'Outside Detection - {detect_type.upper()}', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




