"""HSV 颜色范围。"""

import numpy as np


class HSVColorRegistry:
    def __init__(self):
        self.colors = {
            # 红色位于 HSV 色相轴两端，因此需要两个范围
            "red": [
                ([0, 100, 80], [10, 255, 255]),
                ([170, 100, 80], [179, 255, 255]),
            ],
            "orange": [
                ([7, 100, 100], [20, 255, 255]),
            ],
            "yellow": [
                ([21, 100, 100], [35, 255, 255]),
            ],
            "green": [
                ([36, 70, 70], [85, 255, 255]),
            ],
            "cyan": [
                ([80, 70, 70], [100, 255, 255]),
            ],
            "blue": [
                ([90, 80, 70], [130, 255, 255]),
            ],
            "purple": [
                ([125, 60, 60], [165, 255, 255]),
            ],
            "white": [
                ([0, 0, 200], [179, 55, 255]),
            ],
            "black": [
                ([0, 0, 0], [179, 255, 60]),
            ],
            "gray": [
                ([0, 0, 60], [179, 50, 190]),
            ],
        }

    def get_color_bounds(self, color_name):
        color_name = color_name.lower()

        if color_name not in self.colors:
            supported_colors = ", ".join(self.colors.keys())

            raise ValueError(
                f"不支持颜色 {color_name!r}，"
                f"支持的颜色有：{supported_colors}"
            )

        return [
            (
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
            for lower, upper in self.colors[color_name]
        ]
