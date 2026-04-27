
import cv2
import numpy as np
from collections import Counter
import outsite

def find_type(imgsrc):
    mode = 0
    type_info = []
    blur_contour = imgsrc.copy()
    if mode == 0:
        flag, img, trapezoid_info, center_max, width_max = outsite.detect_trapezoids(imgsrc)
        if flag == 1:
            type_info.append({"type": 1, "center": center_max, "lengh": width_max})
        else:
            mode = 1
    if mode == 1:
        flag, img, triangles_info, center_max, radius_max = outsite.detect_triangle(imgsrc)
        if flag == 1:
            type_info.append({"type": 2, "center": center_max, "lengh": radius_max})
        else:
            mode = 2
    if mode == 2:
        flag, img, pole_groups, center = outsite.find_longest_straight_line(imgsrc)
        if flag == 1:
            type_info.append({"type": 3, "center": center, "lengh": 0})
        else:
            mode = 3
    if mode == 3:
        flag, img, ellipse_info, center, r_max = outsite.detect_ellipses(imgsrc)
        if flag == 1:
            type_info.append({"type": 0, "center": center, "lengh": r_max})
    return img, type_info

def blur_contour_only(src_img, contour, dilate_radius=5, blur_kernel=(25,25)):
    mask = np.zeros_like(src_img[:,:,0])
    cv2.drawContours(mask, [contour], -1, 255, thickness=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_radius+1,)*2)
    expanded_mask = cv2.dilate(mask, kernel)
    blurred = cv2.GaussianBlur(src_img, blur_kernel, 0)
    condition = expanded_mask[:,:,None].astype(bool)
    result = np.where(condition, blurred, src_img)
    return result.astype(np.uint8)

def singel_match(good_match, type, kp0, kp1):
    if len(good_match) < 10:
        return False, 0
    else:
        if type == 0:
            if len(good_match) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                dst_pts = np.float32([kp0[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = mask.ravel().tolist().count(1)
                if inliers/len(good_match) > 0.6:
                    return True, inliers/len(good_match)
                else:
                    return False, inliers/len(good_match)
            return False, 0
        elif type == 1:
            avg_distance = sum([m.distance for m in good_match]) / len(good_match)
            if avg_distance < 50:
                return True, avg_distance
            else:
                return False, avg_distance
        else:
            print("模式错误")
            return False, 0

def match_indicater(good_match, type, kp0, kp1):
    if len(good_match) < 10:
        return False
    else:
        if type == 0:
            if len(good_match) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                dst_pts = np.float32([kp0[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = mask.ravel().tolist().count(1)
                if inliers/len(good_match) > 0.7:
                    return True
                else:
                    return False
            return False
        elif type == 1:
            avg_distance = sum([m.distance for m in good_match]) / len(good_match)
            if avg_distance < 30:
                return True
            else:
                return False
        else:
            print("模式错误")
            return False