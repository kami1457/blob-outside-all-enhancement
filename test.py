
import cv2
from allin import AllIn

frame = cv2.imread('test.jpg')
if frame is None:
    print("无法读取图片，请检查路径")
    exit()

detector1 = AllIn(mode='color_shape', color='red', bais=5)
result1, info1 = detector1.process(frame)
print("颜色+形状识别结果:", info1)
cv2.imshow('color_shape', result1)

detector2 = AllIn(mode='multi_color', color1='red', color2='blue')
result2, info2 = detector2.process(frame)
cv2.imshow('multi_color', result2)

detector3 = AllIn(mode='line', color='red')
result3, info3 = detector3.process(frame)
cv2.imshow('line', result3)

detector4 = AllIn(mode='laser', color='red', bais=5, min_area=500)
result4, info4 = detector4.process(frame)
cv2.imshow('laser', result4)

template = cv2.imread('template.jpg')
if template is not None:
    detector5 = AllIn(mode='orb', color='red', template=template)
    result5, info5 = detector5.process(frame)
    print("ORB 匹配结果:", info5)
    cv2.imshow('orb', result5)

cv2.waitKey(0)
cv2.destroyAllWindows()