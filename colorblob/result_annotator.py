"""检测结果绘制。"""

import cv2


class ResultAnnotator:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(
        self,
        frame,
        contours,
        target_color,
        fps,
    ):
        result = frame.copy()

        for index, contour in enumerate(contours, start=1):
            area = cv2.contourArea(contour)

            x, y, width, height = cv2.boundingRect(contour)

            # 绘制轮廓
            cv2.drawContours(
                result,
                [contour],
                -1,
                (0, 255, 0),
                2,
            )

            # 绘制外接矩形
            cv2.rectangle(
                result,
                (x, y),
                (x + width, y + height),
                (0, 0, 255),
                2,
            )

            # 计算轮廓中心点
            moments = cv2.moments(contour)

            if moments["m00"] != 0:
                center_x = int(
                    moments["m10"] / moments["m00"]
                )

                center_y = int(
                    moments["m01"] / moments["m00"]
                )

                cv2.circle(
                    result,
                    (center_x, center_y),
                    5,
                    (0, 255, 255),
                    -1,
                )

                cv2.putText(
                    result,
                    f"({center_x}, {center_y})",
                    (center_x + 8, center_y),
                    self.font,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            label = (
                f"{index}: {target_color.upper()} "
                f"AREA={int(area)}"
            )

            text_y = max(55, y - 8)

            cv2.putText(
                result,
                label,
                (x, text_y),
                self.font,
                0.5,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 顶部状态栏
        cv2.rectangle(
            result,
            (0, 0),
            (result.shape[1], 38),
            (0, 0, 0),
            -1,
        )

        status_text = (
            f"TARGET: {target_color.upper()}   "
            f"OBJECTS: {len(contours)}   "
            f"FPS: {fps:.1f}"
        )

        cv2.putText(
            result,
            status_text,
            (10, 25),
            self.font,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return result
