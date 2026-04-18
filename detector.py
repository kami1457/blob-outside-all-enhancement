import cv2
import numpy as np


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
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_bounds:
            current_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, current_mask)
        return mask


class MorphologicalOperation:
    def __init__(self, kernel_size=(5, 5)):
        self.kernel = np.ones(kernel_size, np.uint8)

    def apply(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
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