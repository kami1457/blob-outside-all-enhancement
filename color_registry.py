class HSVColorRegistry:
    def __init__(self):
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
        color_name = color_name.lower()
        if color_name in self.colors:
            return self.colors[color_name]
        else:
            print(f"警告: 不支持的颜色 '{color_name}'。支持的颜色有: {self.available_colors()}")
            return None

    def available_colors(self):
        return list(self.colors.keys())

    def add_custom_color(self, name, bounds_list):
        self.colors[name.lower()] = bounds_list