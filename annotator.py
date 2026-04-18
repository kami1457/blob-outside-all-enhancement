import cv2


class ResultAnnotator:
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def annotate(self, image, contours, target_color):
        annotated_image = image.copy()

        for contour in contours:
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(annotated_image, [approx], 0, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(contour)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(annotated_image, (cx, cy), 4, (0, 255, 255), -1)

            label = f"{target_color.upper()}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )

            bg_y = max(0, y - text_h - 10)

            cv2.rectangle(
                annotated_image,
                (x, bg_y),
                (x + text_w, bg_y + text_h + 10),
                (255, 0, 0),
                -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x, bg_y + text_h + 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )

        return annotated_image