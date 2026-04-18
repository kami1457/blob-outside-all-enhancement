import os
import cv2

from color_registry import HSVColorRegistry
from detector import ColorBlobDetector
from annotator import ResultAnnotator


def main():
    color_registry = HSVColorRegistry()
    annotator = ResultAnnotator()

    target_color = 'red'
    image_path = r"D:\Users\pengy\Downloads\chatgpt.png"

    color_bounds = color_registry.get_color_bounds(target_color)
    if color_bounds is None:
        return

    if not os.path.isfile(image_path):
        print(f"无法找到图像文件: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        return

    detector = ColorBlobDetector(color_bounds, resize_factor=0.5, min_area=800)
    processed_image, contours, mask = detector.detect(image)

    final_image = annotator.annotate(processed_image, contours, target_color)

    cv2.imshow(f'Detected {target_color.upper()} Blobs', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()