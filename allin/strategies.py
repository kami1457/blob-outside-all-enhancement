
import cv2
import numpy as np
from collections import Counter

import colorblob
import outsite   

from utils import find_type, blur_contour_only, singel_match
from strategy_base import BaseStrategy

@BaseStrategy.register_strategy('color_shape')
class ColorShapeStrategy(BaseStrategy):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = self.params.get('color', 'red')
        self.bais = self.params.get('bais', 0)

    def process(self, frame):
        import time
        start_time = time.time()
        self._validate_frame(frame)
        
        cached_result = self._get_from_cache(frame)
        if cached_result:
            self._logger.info(f'从缓存获取结果，处理时间: {time.time() - start_time:.4f}s')
            return cached_result
        
        cnt_efc = colorblob.detect_color(frame, self.color, self.bais)
        type_list = []

        for item in cnt_efc:
            result_img = item['result']
            result_img, type_info = find_type(result_img)
            item['result'] = result_img
            type_list.extend(type_info)

        composite_img = self._compose_images(frame, cnt_efc)
        result = (composite_img, type_list)
        
        self._add_to_cache(frame, result)
        
        processing_time = time.time() - start_time
        self._logger.info(f'处理完成，识别到 {len(type_list)} 个目标，处理时间: {processing_time:.4f}s')
        return result

@BaseStrategy.register_strategy('orb')
class ORBStrategy(BaseStrategy):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = self.params.get('color', 'red')
        self.template = self.params.get('template', None)
        self._template_kp = None
        self._template_des = None
        self._orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def process(self, frame):
        import time
        start_time = time.time()
        self._validate_frame(frame)
        if self.template is None:
            raise ValueError("ORBStrategy 需要提供 template 参数")
        
        cached_result = self._get_from_cache(frame)
        if cached_result:
            self._logger.info(f'从缓存获取结果，处理时间: {time.time() - start_time:.4f}s')
            return cached_result
        
        composite = self._prepare_orb_image(frame)
        result_img, info = self._template_match(composite)
        result = (result_img, info)
        
        self._add_to_cache(frame, result)
        
        processing_time = time.time() - start_time
        self._logger.info(f'ORB匹配完成，成功: {info.get("success", False)}，得分: {info.get("score", 0):.2f}，处理时间: {processing_time:.4f}s')
        return result

    def _prepare_orb_image(self, frame):
        cnt_efc = colorblob.detect_color(frame, self.color, bais=0)

        for item in cnt_efc:
            result_img = item['result']
            test_img, contour = outsite.detect_ellipse_max_one(result_img)
            if contour is not None:
                result_img = blur_contour_only(result_img, contour, dilate_radius=5, blur_kernel=(25, 25))
            item['result'] = result_img

        composite_img = self._compose_images(frame, cnt_efc)
        return composite_img

    def _template_match(self, candidate):
        
        if self._template_kp is None or self._template_des is None:
            
            h, w = self.template.shape[:2]
            aspect_ratio = w / h
            if w > h:
                new_w = 640
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = 480
                new_w = int(new_h * aspect_ratio)
            self._resized_template = cv2.resize(self.template, (new_w, new_h))
            self._template_kp, self._template_des = self._orb.detectAndCompute(self._resized_template, None)
        else:
            self._resized_template = cv2.resize(self.template, (self._resized_template.shape[1], self._resized_template.shape[0]))

        candidate = cv2.resize(candidate, (self._resized_template.shape[1], self._resized_template.shape[0]))

        kp1, des1 = self._orb.detectAndCompute(candidate, None)

        good_match = []
        if des1 is not None and self._template_des is not None:
            matches = self._bf.match(des1, self._template_des)
            matches = [m for m in matches if m.distance < 50]
            good_match = sorted(matches, key=lambda x: x.distance)[:45]

        outimage = cv2.drawMatches(candidate, kp1, self._resized_template, self._template_kp, good_match, outImg=None)
        indicater, score = singel_match(good_match, 1, self._template_kp, kp1)

        info = {'success': indicater, 'score': score}
        return outimage, info

@BaseStrategy.register_strategy('line')
class LineStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = self.params.get('color', 'red')

    def process(self, frame):
        import time
        start_time = time.time()
        self._validate_frame(frame)
        
        cached_result = self._get_from_cache(frame)
        if cached_result:
            self._logger.info(f'从缓存获取结果，处理时间: {time.time() - start_time:.4f}s')
            return cached_result
        
        cnt_efc = colorblob.detect_color_to_rect(frame, self.color)

        for item in cnt_efc:
            result_img = item['result']
            flag, result_img, pole_groups, center = outsite.find_longest_straight_line(result_img)
            item['result'] = result_img

        composite_img = self._compose_images(frame, cnt_efc)
        result = (composite_img, [])
        
        self._add_to_cache(frame, result)
        
        processing_time = time.time() - start_time
        self._logger.info(f'直线检测完成，处理时间: {processing_time:.4f}s')
        return result

@BaseStrategy.register_strategy('multi_color')
class MultiColorStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color1 = self.params.get('color1', 'red')
        self.color2 = self.params.get('color2', 'blue')

    def process(self, frame):
        import time
        start_time = time.time()
        self._validate_frame(frame)
        
        cached_result = self._get_from_cache(frame)
        if cached_result:
            self._logger.info(f'从缓存获取结果，处理时间: {time.time() - start_time:.4f}s')
            return cached_result
        
        cnt_efc = colorblob.detect_multi_color(frame, self.color1, self.color2)
        composite_img = self._compose_images(frame, cnt_efc)
        result = (composite_img, [])
        
        self._add_to_cache(frame, result)
        
        processing_time = time.time() - start_time
        self._logger.info(f'多颜色合成完成，处理时间: {processing_time:.4f}s')
        return result

@BaseStrategy.register_strategy('laser')
class LaserStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = self.params.get('color', 'red')
        self.bais = self.params.get('bais', 5)
        self.min_area = self.params.get('min_area', 500)

    def process(self, frame):
        import time
        start_time = time.time()
        self._validate_frame(frame)
        
        cached_result = self._get_from_cache(frame)
        if cached_result:
            self._logger.info(f'从缓存获取结果，处理时间: {time.time() - start_time:.4f}s')
            return cached_result
        
        cnt_efc = colorblob.detect_color(frame, self.color, bais=self.bais, min_area=self.min_area)
        composite_img = self._compose_images(frame, cnt_efc)
        result = (composite_img, [])
        
        self._add_to_cache(frame, result)
        
        processing_time = time.time() - start_time
        self._logger.info(f'激光检测完成，处理时间: {processing_time:.4f}s')
        return result