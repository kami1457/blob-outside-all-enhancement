"""程序入口。"""

import threading

import cv2

from camera_worker import camera_worker
from config import MIN_AREA, TARGET_COLOR


def main():
    cv2.setUseOptimized(True)

    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(stop_event,),
        daemon=True,
    )

    camera_thread.start()

    print()
    print("====================================")
    print("树莓派颜色识别程序已经启动。")
    print("====================================")
    print(f"识别颜色：{TARGET_COLOR}")
    print(f"最小面积：{MIN_AREA}")
    print()
    print("按 Ctrl+C 停止程序。")
    print("====================================")
    print()

    try:
        while camera_thread.is_alive():
            camera_thread.join(timeout=0.5)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止程序。")

    finally:
        stop_event.set()

        camera_thread.join(timeout=3.0)

        print("程序已结束。")


if __name__ == "__main__":
    main()
