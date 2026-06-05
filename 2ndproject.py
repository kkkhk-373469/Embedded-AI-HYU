from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera
from ultralytics import YOLO
import cv2
import time
import numpy as np

# ============================================================
# Left Camera + left_start.engine PID Road Following
# Stop when circle bbox width >= 280 px
# ============================================================

# =====================================
# Vehicle connection
# =====================================
base = BaseController('/dev/ttyUSB0', 115200)

# =====================================
# Camera
# CAM1 = left camera
# CAM0 = right camera
# =====================================
camera = CSICamera(
    capture_device=1,       # CAM1 = 왼쪽 카메라
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)

# =====================================
# Models
# =====================================
seg_model = YOLO(
    'left_start.engine',
    task='segment'
)

circle_model = YOLO(
    'best_circle.pt'
)

# =====================================
# Circle detection condition
# =====================================
CIRCLE_CONF = 0.60
CIRCLE_MIN_WIDTH = 280

# 감지 안정성을 위해 연속 프레임 조건을 쓰고 싶으면 2~3으로 올리면 됨
# 사용자 요구: 280px 이상 감지하면 멈춤 → 기본 1프레임
CIRCLE_CONFIRM_FRAMES = 1

# =====================================
# Motor / control parameters
# =====================================
MAX_SPEED = 0.50
MAX_STEER = 0.79

BASE_SPEED = 0.20
TURN_STRENGTH = 0.28

# 예전에 쓰던 코드처럼 실제 전송 시 L/R을 반대로 보내야 하는 경우 True
SWAP_LR_COMMAND = True

# segmentation mask가 잠깐 안 잡혔을 때 이전 steering 유지
MAX_MASK_HOLD_FRAMES = 5

# PID parameters
# # error_norm = (lane_center - image_center) / image_center
# steering = Kp*e + Ki*integral + Kd*derivative
PID_KP = 1.15
PID_KI = 0.00
PID_KD = 0.18

# integral이 너무 커지는 것 방지
INTEGRAL_LIMIT = 0.50

# steering smoothing
STEERING_SMOOTHING = 0.35

# ROI: mask 하단 몇 %를 사용할지
ROI_START_RATIO = 0.70


class PIDController:
    def __init__(self, kp, ki, kd, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time

        if dt <= 1e-6:
            dt = 1e-3

        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )

        derivative = (error - self.prev_error) / dt

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.prev_error = error
        self.prev_time = now

        return output, dt, derivative


pid = PIDController(
    PID_KP,
    PID_KI,
    PID_KD,
    INTEGRAL_LIMIT
)


def clip(value, max_value):
    return max(
        min(value, max_value),
        -max_value
    )


def send_motor(logical_L, logical_R):
    """
    logical_L / logical_R:
        제어 로직 기준 왼쪽/오른쪽 바퀴 명령

    실제 하드웨어가 예전 코드처럼 L/R 반대로 들어가야 하면
    SWAP_LR_COMMAND=True로 두고 base_json_ctrl에는 R, L 순서로 보냄.
    """

    logical_L = clip(logical_L, MAX_SPEED)
    logical_R = clip(logical_R, MAX_SPEED)

    if SWAP_LR_COMMAND:
        send_L = logical_R
        send_R = logical_L
    else:
        send_L = logical_L
        send_R = logical_R

    base.base_json_ctrl({
        'T': 1,
        'L': float(send_L),
        'R': float(send_R)
    })

    return send_L, send_R


def stop_vehicle():
    base.base_json_ctrl({
        'T': 1,
        'L': 0.0,
        'R': 0.0
    })


def warmup_models():
    dummy_frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8
    )

    print("[WARMUP] left_start.engine")
    seg_model.predict(
        source=dummy_frame,
        imgsz=640,
        verbose=False
    )

    print("[WARMUP] best_circle.pt")
    circle_model.predict(
        source=dummy_frame,
        imgsz=640,
        verbose=False
    )

    print("[WARMUP] Done")


def detect_circle(frame):
    """
    circle bbox width가 CIRCLE_MIN_WIDTH 이상이면 True.
    """

    results = circle_model.predict(
        source=frame,
        conf=CIRCLE_CONF,
        imgsz=640,
        verbose=False
    )

    max_width = 0.0
    max_conf = 0.0

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = str(circle_model.names[cls_id]).lower()
            conf = float(box.conf[0].item())

            if label != "circle":
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            box_width = float(x2 - x1)

            if box_width > max_width:
                max_width = box_width
                max_conf = conf

    if max_width >= CIRCLE_MIN_WIDTH:
        print(
            f"[CIRCLE] DETECTED "
            f"conf={max_conf:.2f} "
            f"width={max_width:.1f}px "
            f">= {CIRCLE_MIN_WIDTH}px"
        )
        return True, max_width, max_conf

    return False, max_width, max_conf


def get_steering_from_segmentation(frame, prev_steering):
    """
    left_start.engine segmentation mask 기반으로 lane center 계산.
    PID 제어를 이용해 steering 출력.
    """

    results = seg_model.predict(
        source=frame,
        imgsz=640,
        verbose=False
    )

    if len(results) == 0:
        return prev_steering, False, None

    r = results[0]

    if (
        r.masks is None
        or r.masks.data is None
        or r.masks.data.shape[0] == 0
    ):
        return prev_steering, False, None

    masks = (
        r.masks.data
        .cpu()
        .numpy()
    )

    # 여러 mask가 잡히면 모두 합침
    mask = (
        np.max(masks, axis=0) > 0.5
    ).astype(np.uint8)

    # 작은 노이즈 제거
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_h, mask_w = mask.shape

    roi = mask[
        int(mask_h * ROI_START_RATIO):,
        :
    ]

    ys, xs = np.where(
        roi > 0
    )

    if len(xs) == 0:
        return prev_steering, False, None

    lane_center = np.mean(xs)
    image_center = mask_w / 2.0

    error = lane_center - image_center
    error_norm = error / image_center

    raw_steering, dt, derivative = pid.update(error_norm)

    # smoothing
    steering = (
        STEERING_SMOOTHING * prev_steering
        + (1.0 - STEERING_SMOOTHING) * raw_steering
    )

    steering = clip(
        steering,
        MAX_STEER
    )

    debug_info = {
        "lane_center": lane_center,
        "image_center": image_center,
        "error_norm": error_norm,
        "raw_steering": raw_steering,
        "dt": dt,
        "derivative": derivative
    }

    return steering, True, debug_info


def update_vehicle_motion(steering, speed):
    """
    steering > 0이면 기존 코드 기준으로:
        L = speed - steering * TURN_STRENGTH
        R = speed + steering * TURN_STRENGTH
    """

    steering = clip(
        steering,
        MAX_STEER
    )

    logical_L = speed - steering * TURN_STRENGTH
    logical_R = speed + steering * TURN_STRENGTH

    sent_L, sent_R = send_motor(
        logical_L,
        logical_R
    )

    print(
        f"[AUTO] "
        f"steer={steering:+.3f} "
        f"logical_L={logical_L:+.3f} "
        f"logical_R={logical_R:+.3f} "
        f"sent_L={sent_L:+.3f} "
        f"sent_R={sent_R:+.3f}"
    )


def main():
    print("==============================================")
    print(" Left Camera PID Road Following")
    print(" Camera: CAM1")
    print(" Segmentation: left_start.engine")
    print(f" Stop condition: circle width >= {CIRCLE_MIN_WIDTH}px")
    print(f" SWAP_LR_COMMAND = {SWAP_LR_COMMAND}")
    print("==============================================")

    warmup_models()

    prev_steering = 0.0
    mask_miss_count = 0
    circle_count = 0

    stop_vehicle()
    time.sleep(1.0)

    try:
        while True:
            frame = camera.read()

            if frame is None:
                continue

            # =====================================
            # 1. Circle detection
            # =====================================
            circle_detected, circle_width, circle_conf = detect_circle(frame)

            if circle_detected:
                circle_count += 1
            else:
                circle_count = 0

            if circle_count >= CIRCLE_CONFIRM_FRAMES:
                print(
                    f"[STOP] Circle detected for "
                    f"{circle_count} frame(s). Stop vehicle."
                )
                stop_vehicle()
                break

            # =====================================
            # 2. Segmentation + PID steering
            # =====================================
            steering, valid_mask, debug = get_steering_from_segmentation(
                frame,
                prev_steering
            )

            if valid_mask:
                prev_steering = steering
                mask_miss_count = 0

                print(
                    f"[PID] "
                    f"error={debug['error_norm']:+.3f} "
                    f"raw={debug['raw_steering']:+.3f} "
                    f"steer={steering:+.3f}"
                )

            else:
                mask_miss_count += 1

                if mask_miss_count <= MAX_MASK_HOLD_FRAMES:
                    steering = prev_steering
                    print(
                        f"[MASK MISS] hold prev steering "
                        f"{mask_miss_count}/{MAX_MASK_HOLD_FRAMES}"
                    )
                else:
                    # 너무 오래 mask가 안 잡히면 안전하게 감속/정지
                    steering = 0.0
                    print("[MASK LOST] stop for safety")
                    stop_vehicle()
                    continue

            # =====================================
            # 3. Motor control
            # =====================================
            update_vehicle_motion(
                steering,
                BASE_SPEED
            )

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[USER STOP] KeyboardInterrupt")

    finally:
        stop_vehicle()
        cv2.destroyAllWindows()
        print("[SAFE] Motors stopped.")


if __name__ == "__main__":
    main()
