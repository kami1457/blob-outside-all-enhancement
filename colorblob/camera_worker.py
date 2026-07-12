"""摄像头处理子程序。"""

import time

import cv2
import numpy as np
from picamera2 import Picamera2

from color_blob_detector import ColorBlobDetector
from config import CAMERA_FPS, FRAME_HEIGHT, FRAME_WIDTH, MIN_AREA, TARGET_COLOR
from hsv_color_registry import HSVColorRegistry
from result_annotator import ResultAnnotator


def camera_worker(stop_event):
    registry = HSVColorRegistry()

    color_bounds = registry.get_color_bounds(
        TARGET_COLOR
    )

    detector = ColorBlobDetector(
        color_bounds=color_bounds,
        min_area=MIN_AREA,
    )

    annotator = ResultAnnotator()

    picam2 = Picamera2()
    camera_started = False

    try:
        camera_config = picam2.create_video_configuration(
            main={
                # 这里必须使用 RGB888
                # 使用 BGR888 会导致红蓝通道交换
                "format": "RGB888",

                "size": (
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                ),
            },

            controls={
                "FrameRate": CAMERA_FPS,
            },

            buffer_count=6,
        )

        picam2.configure(camera_config)
        picam2.start()

        camera_started = True

        # 等待自动曝光和自动白平衡稳定
        time.sleep(2.0)

        print("摄像头启动成功。")
        print(
            f"分辨率：{FRAME_WIDTH}x{FRAME_HEIGHT}"
        )
        print(
            f"目标帧率：{CAMERA_FPS:.0f}"
        )

        frame_count = 0
        current_fps = 0.0
        fps_start_time = time.monotonic()

        while not stop_event.is_set():
            frame = picam2.capture_array("main")

            if frame is None or frame.size == 0:
                print("警告：摄像头返回空画面。")
                time.sleep(0.05)
                continue

            # 镜像显示
            frame = cv2.flip(frame, 1)

            contours, mask = detector.detect(frame)

            # FPS 计算
            frame_count += 1

            current_time = time.monotonic()
            elapsed_time = current_time - fps_start_time

            if elapsed_time >= 1.0:
                current_fps = (
                    frame_count / elapsed_time
                )

                frame_count = 0
                fps_start_time = current_time

            annotated_frame = annotator.annotate(
                frame,
                contours,
                TARGET_COLOR,
                current_fps,
            )

            # 单通道掩膜转为三通道
            mask_display = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR,
            )

            # 掩膜顶部状态栏
            cv2.rectangle(
                mask_display,
                (0, 0),
                (mask_display.shape[1], 38),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                mask_display,
                "HSV BINARY MASK",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 左右拼接
            np.hstack(
                (
                    annotated_frame,
                    mask_display,
                )
            )

    except Exception as error:
        print(
            "\n摄像头处理发生错误："
            f"{type(error).__name__}: {error}"
        )

        stop_event.set()

    finally:
        if camera_started:
            try:
                picam2.stop()
            except Exception:
                pass

        try:
            picam2.close()
        except Exception:
            pass

        print("摄像头已关闭。")
