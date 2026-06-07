from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np

# =====================================
# Vehicle connection
# =====================================
base = BaseController('/dev/ttyUSB0', 115200)


def resolve_first_existing_path(model_names):
    for model_name in model_names:
        if os.path.exists(model_name):
            return model_name

    return None

# =====================================
# Cameras
# CAM1 = left camera
# CAM0 = right camera
# =====================================
left_camera = CSICamera(
    capture_device=1,       # CAM1 = 왼쪽 카메라
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)

right_camera = CSICamera(
    capture_device=0,       # CAM0 = 오른쪽 카메라
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)

# =====================================
# Models
# =====================================
left_seg_model = YOLO(
    'left_start.engine',
    task='segment'
)

right_seg_model = YOLO(
    'right.engine',
    task='segment'
)

circle_model = YOLO(
    'best_circle.pt'
)

sign_model = YOLO(
    'sign.pt'
)

tire_model = YOLO(
    'tire.pt'
)

LANE_V2_MODEL_PATH = resolve_first_existing_path([
    'lane_v3.pt'
])

lane_v2_model = (
    YOLO(LANE_V2_MODEL_PATH)
    if LANE_V2_MODEL_PATH is not None
    else None
)

LEFT_CAMERA_CENTER = 780
RIGHT_CAMERA_CENTER = 500

TIRE_CONF = 0.6

# =====================================
# Circle detection condition
# =====================================
CIRCLE_CONF = 0.65
CIRCLE_MIN_WIDTH = 280

# 감지 안정성을 위해 연속 프레임 조건을 쓰고 싶으면 2~3으로 올리면 됨
# 사용자 요구: 280px 이상 감지하면 멈춤 → 기본 1프레임
CIRCLE_CONFIRM_FRAMES = 2

# =====================================
# Sign detection before circle stop
# =====================================
SIGN_CONF = 0.50
SIGN_CONFIRM_FRAMES = 3
SIGN_LABEL_TO_RIGHT_BOX_COUNT = {
    "left": 3,
    "right": 1,
    "u_turn": 4,
    "go_ahead": 2
}

# =====================================
# Forced right steering after sign stop
# =====================================
SIGN_STOP_SECONDS = 0.5
FORCE_RIGHT_SECONDS = 1.8
POST_FORCE_STOP_SECONDS = 1.4
FORCE_RIGHT_SPEED = 0.23
FORCE_RIGHT_STEERING = 0.28

# =====================================
# Lane v2 bbox following after forced right steering
# =====================================
LANE_V2_CONF = 0.30
LANE_V2_ALLOWED_LABELS = {"right-left-lane"}
LANE_V2_RIGHT_BOX_CONF = 0.3
LANE_V2_MAX_BOXES = 2
LANE_V2_DIVERGE_DELTA_PX = 600.0
LANE_V2_TREND_EPS_PX = 10.0
LANE_V2_TREND_CONFIRM_FRAMES = 2
LANE_V2_RIGHT_BOX_COUNT = 4

# =====================================
# Motor / control parameters
# =====================================
MAX_SPEED = 0.50
MAX_STEER = 0.9

BASE_SPEED = 0.20
TURN_STRENGTH = 0.26

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

# 오른쪽 카메라 조향 부호 보정.
LEFT_ERROR_SIGN = 1.0
RIGHT_ERROR_SIGN = 1.0

# 양쪽 카메라가 모두 유효할 때 error_norm 가중 평균 비율
LEFT_ERROR_WEIGHT = 0.50
RIGHT_ERROR_WEIGHT = 0.50


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


def make_lane_gap_state():
    return {
        "prev_gap": None,
        "min_gap": None,
        "trend": "unknown",
        "decrease_streak": 0,
        "increase_streak": 0,
        "diverge_active": False
    }


def warmup_models():
    dummy_frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8
    )

    print("[WARMUP] left_start.engine")
    left_seg_model.predict(
        source=dummy_frame,
        imgsz=640,
        verbose=False
    )

    print("[WARMUP] right.engine")
    right_seg_model.predict(
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

    print("[WARMUP] sign.pt")
    sign_model.predict(
        source=dummy_frame,
        imgsz=640,
        verbose=False
    )

    if lane_v2_model is not None:
        print(f"[WARMUP] {LANE_V2_MODEL_PATH}")
        lane_v2_model.predict(
            source=dummy_frame,
            imgsz=640,
            verbose=False
        )
    else:
        print("[WARMUP] lane_v2 model missing, skip")

    print("[WARMUP] Done")


def detect_circle(frame, camera_name):
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
            f"[CIRCLE:{camera_name}] DETECTED "
            f"conf={max_conf:.2f} "
            f"width={max_width:.1f}px "
            f">= {CIRCLE_MIN_WIDTH}px"
        )
        return True, max_width, max_conf

    return False, max_width, max_conf


def detect_circle_dual(left_frame, right_frame):
    """
    양쪽 카메라 중 하나라도 circle bbox width 조건을 만족하면 True.
    """

    left_detected = False
    left_width = 0.0
    left_conf = 0.0
    right_detected = False
    right_width = 0.0
    right_conf = 0.0

    if left_frame is not None:
        left_detected, left_width, left_conf = detect_circle(
            left_frame,
            "left"
        )

    if right_frame is not None:
        right_detected, right_width, right_conf = detect_circle(
            right_frame,
            "right"
        )

    if left_detected or right_detected:
        if left_width >= right_width:
            return True, left_width, left_conf, "left"
        return True, right_width, right_conf, "right"

    if left_width >= right_width:
        return False, left_width, left_conf, "left"
    return False, right_width, right_conf, "right"


def detect_tire(frame, source):

    results = tire_model.predict(
        source=frame,
        conf=TIRE_CONF,
        imgsz=640,
        verbose=False
    )

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1,y1,x2,y2=map(
                int,
                box.xyxy[0]
            )

            conf=float(box.conf[0])

            cx=(x1+x2)/2

            return True,cx,x1,y1,x2,y2,conf,source

    return False,None,None,None,None,None,None,None

def detect_tire_dual(left_frame,right_frame):

    if left_frame is not None:

        ok,cx,x1,y1,x2,y2,conf,_=detect_tire(
            left_frame,
            "LEFT"
        )

        if ok:
            return True,cx,x1,y1,x2,y2,conf,"LEFT"

    if right_frame is not None:

        ok,cx,x1,y1,x2,y2,conf,_=detect_tire(
            right_frame,
            "RIGHT"
        )

        if ok:
            return True,cx,x1,y1,x2,y2,conf,"RIGHT"

    return False,None,None,None,None,None,None,None


def normalize_sign_label(label):
    return str(label).strip().lower().replace("-", "_")


def detect_sign_labels(frame):
    """
    sign.pt에서 conf 0.5 이상인 관심 라벨을 프레임 단위로 반환한다.
    같은 라벨이 한 프레임에 여러 bbox로 잡혀도 1번만 센다.
    """

    results = sign_model.predict(
        source=frame,
        conf=SIGN_CONF,
        imgsz=640,
        verbose=False
    )

    detected_labels = set()

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = normalize_sign_label(sign_model.names[cls_id])

            if label in SIGN_LABEL_TO_RIGHT_BOX_COUNT:
                detected_labels.add(label)

    return detected_labels


def detect_sign_labels_dual(left_frame, right_frame):
    detected_labels = set()

    if left_frame is not None:
        detected_labels.update(
            detect_sign_labels(left_frame)
        )

    if right_frame is not None:
        detected_labels.update(
            detect_sign_labels(right_frame)
        )

    return detected_labels


def choose_lane_v2_right_box_count(sign_counts, default_count):
    valid_counts = {
        label: count
        for label, count in sign_counts.items()
        if count >= SIGN_CONFIRM_FRAMES
    }

    if len(valid_counts) == 0:
        return default_count, None, 0

    best_label = max(
        valid_counts,
        key=lambda label: valid_counts[label]
    )

    return (
        SIGN_LABEL_TO_RIGHT_BOX_COUNT[best_label],
        best_label,
        valid_counts[best_label]
    )


def get_top_lane_v2_boxes(frame):
    """
    lane_v3 detection 결과 중 right-left-lane bbox만 추출하고,
    confidence 상위 2개 bbox까지만 사용한다.
    반환 bbox는 화면 x좌표 기준 왼쪽에서 오른쪽 순서로 정렬된다.
    """

    if lane_v2_model is None:
        print("[LANE_V2] model file missing: lane_v3.pt")
        return []

    results = lane_v2_model.predict(
        source=frame,
        conf=LANE_V2_CONF,
        imgsz=640,
        verbose=False
    )

    lane_boxes = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = str(lane_v2_model.names[cls_id]).lower()

            if label not in LANE_V2_ALLOWED_LABELS:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)

            lane_boxes.append({
                "conf": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": center_x,
                "center_y": center_y
            })

    lane_boxes = sorted(
        lane_boxes,
        key=lambda lane_box: lane_box["conf"],
        reverse=True
    )[:LANE_V2_MAX_BOXES]

    return sorted(
        lane_boxes,
        key=lambda lane_box: lane_box["center_x"]
    )


def get_steering_from_lane_v2_boxes(
    left_frame,
    right_frame,
    prev_steering,
    diverge_count,
    gap_state
):
    frame = left_frame
    camera_name = "left"

    if frame is None:
        return (
            prev_steering,
            False,
            diverge_count,
            gap_state,
            None
        )

    lane_boxes = get_top_lane_v2_boxes(frame)

    if len(lane_boxes) == 0:
        return (
            prev_steering,
            False,
            diverge_count,
            gap_state,
            None
        )

    current_pair_gap = None
    diverge_threshold = None

    if len(lane_boxes) >= 2:
        left_box = lane_boxes[0]
        right_box = lane_boxes[1]
        current_pair_gap = right_box["center_x"] - left_box["center_x"]

        if gap_state["min_gap"] is None:
            gap_state["min_gap"] = current_pair_gap
        elif (
            not gap_state["diverge_active"]
            and current_pair_gap < gap_state["min_gap"]
        ):
            gap_state["min_gap"] = current_pair_gap

        prev_gap = gap_state["prev_gap"]

        if prev_gap is not None:
            if current_pair_gap < prev_gap - LANE_V2_TREND_EPS_PX:
                gap_state["decrease_streak"] += 1
                gap_state["increase_streak"] = 0

                if gap_state["decrease_streak"] >= LANE_V2_TREND_CONFIRM_FRAMES:
                    if gap_state["trend"] != "decreasing":
                        gap_state["min_gap"] = current_pair_gap
                    else:
                        gap_state["min_gap"] = min(
                            gap_state["min_gap"],
                            current_pair_gap
                        )

                    gap_state["trend"] = "decreasing"
                    gap_state["diverge_active"] = False

            elif current_pair_gap > prev_gap + LANE_V2_TREND_EPS_PX:
                gap_state["increase_streak"] += 1
                gap_state["decrease_streak"] = 0

                if (
                    gap_state["increase_streak"]
                    >= LANE_V2_TREND_CONFIRM_FRAMES
                ):
                    gap_state["trend"] = "increasing"

            else:
                gap_state["decrease_streak"] = 0
                gap_state["increase_streak"] = 0

        if gap_state["trend"] == "decreasing":
            gap_state["min_gap"] = min(
                gap_state["min_gap"],
                current_pair_gap
            )

        diverge_threshold = gap_state["min_gap"] + LANE_V2_DIVERGE_DELTA_PX

        if (
            gap_state["trend"] == "increasing"
            and current_pair_gap >= diverge_threshold
            and not gap_state["diverge_active"]
            and diverge_count < LANE_V2_RIGHT_BOX_COUNT
        ):
            diverge_count += 1
            gap_state["diverge_active"] = True

        gap_state["prev_gap"] = current_pair_gap

    if diverge_count >= LANE_V2_RIGHT_BOX_COUNT:
        right_candidates = [
            lane_box for lane_box in lane_boxes
            if lane_box["conf"] >= LANE_V2_RIGHT_BOX_CONF
        ]

        if len(right_candidates) == 0:
            return (
                prev_steering,
                False,
                diverge_count,
                gap_state,
                None
            )

        right_box = max(
            right_candidates,
            key=lambda lane_box: lane_box["center_x"]
        )
        target_x = right_box["center_x"]
        target_conf = right_box["conf"]
        target_mode = "right_box"

    else:
        left_box = lane_boxes[0]
        target_x = left_box["center_x"]
        target_conf = left_box["conf"]
        target_mode = "left_box"

    image_center = frame.shape[1] / 2.0
    error_norm = (target_x - image_center) / image_center

    raw_steering, dt, derivative = pid.update(error_norm)

    steering = (
        STEERING_SMOOTHING * prev_steering
        + (1.0 - STEERING_SMOOTHING) * raw_steering
    )

    steering = clip(
        steering,
        MAX_STEER
    )

    debug_info = {
        "camera": camera_name,
        "box_count": len(lane_boxes),
        "target_mode": target_mode,
        "target_x": target_x,
        "target_conf": target_conf,
        "image_center": image_center,
        "error_norm": error_norm,
        "raw_steering": raw_steering,
        "dt": dt,
        "derivative": derivative,
        "pair_gap": current_pair_gap,
        "min_pair_gap": gap_state["min_gap"],
        "diverge_threshold": diverge_threshold,
        "gap_trend": gap_state["trend"],
        "decrease_streak": gap_state["decrease_streak"],
        "increase_streak": gap_state["increase_streak"],
        "diverge_active": gap_state["diverge_active"],
        "diverge_count": diverge_count,
        "boxes": lane_boxes
    }

    return (
        steering,
        True,
        diverge_count,
        gap_state,
        debug_info
    )


def get_error_from_segmentation(frame, model, camera_name, error_sign):
    """
    segmentation mask 기반으로 lane center와 image center 차이를 계산.
    PID는 양쪽 카메라 error를 합친 뒤 한 번만 적용한다.
    """

    results = model.predict(
        source=frame,
        imgsz=640,
        verbose=False
    )

    if len(results) == 0:
        return None, False

    r = results[0]

    if (
        r.masks is None
        or r.masks.data is None
        or r.masks.data.shape[0] == 0
    ):
        return None, False

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
        return None, False

    lane_center = np.mean(xs)
    image_center = mask_w / 2.0

    error = lane_center - image_center
    error_norm = (error / image_center) * error_sign

    debug_info = {
        "camera": camera_name,
        "lane_center": lane_center,
        "image_center": image_center,
        "error_norm": error_norm
    }

    return debug_info, True


def fuse_dual_errors(left_debug, left_valid, right_debug, right_valid):
    """
    양쪽 카메라 error_norm을 하나의 PID 입력값으로 합친다.
    한쪽만 유효하면 해당 카메라만 사용한다.
    """

    if left_valid and right_valid:
        total_weight = LEFT_ERROR_WEIGHT + RIGHT_ERROR_WEIGHT
        if total_weight <= 1e-6:
            total_weight = 1.0

        error_norm = (
            left_debug["error_norm"] * LEFT_ERROR_WEIGHT
            + right_debug["error_norm"] * RIGHT_ERROR_WEIGHT
        ) / total_weight

        return error_norm, "dual"

    if left_valid:
        return left_debug["error_norm"], "left"

    if right_valid:
        return right_debug["error_norm"], "right"

    return None, "none"


def get_steering_from_dual_segmentation(
    left_frame,
    right_frame,
    prev_steering
):
    left_debug = None
    left_valid = False
    right_debug = None
    right_valid = False

    if left_frame is not None:
        left_debug, left_valid = get_error_from_segmentation(
            left_frame,
            left_seg_model,
            "left",
            LEFT_ERROR_SIGN
        )

    if right_frame is not None:
        right_debug, right_valid = get_error_from_segmentation(
            right_frame,
            right_seg_model,
            "right",
            RIGHT_ERROR_SIGN
        )

    error_norm, source = fuse_dual_errors(
        left_debug,
        left_valid,
        right_debug,
        right_valid
    )

    if error_norm is None:
        return prev_steering, False, None

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
        "source": source,
        "left": left_debug,
        "right": right_debug,
        "error_norm": error_norm,
        "raw_steering": raw_steering,
        "dt": dt,
        "derivative": derivative
    }

    return steering, True, debug_info


def get_steering_from_right_segmentation(
    right_frame,
    prev_steering
):
    right_debug = None
    right_valid = False

    if right_frame is not None:
        right_debug, right_valid = get_error_from_segmentation(
            right_frame,
            right_seg_model,
            "right",
            RIGHT_ERROR_SIGN
        )

    if right_valid:
        error_norm = right_debug["error_norm"]
        source = "right"
    else:
        error_norm = None
        source = "none"

    if error_norm is None:
        return prev_steering, False, None

    raw_steering, dt, derivative = pid.update(error_norm)

    steering = (
        STEERING_SMOOTHING * prev_steering
        + (1.0 - STEERING_SMOOTHING) * raw_steering
    )

    steering = clip(
        steering,
        MAX_STEER
    )

    debug_info = {
        "source": source,
        "left": None,
        "right": right_debug,
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
    global LANE_V2_RIGHT_BOX_COUNT

    print("==============================================")
    print(" Dual Camera PID Road Following")
    print(" Cameras: CAM1(left), CAM0(right)")
    print(" Segmentation: left_start.engine + right.engine")
    print(" Lane v2 bbox following after forced right steering")
    print(f" Sign stop: circle width >= {CIRCLE_MIN_WIDTH}px")
    print(f" SWAP_LR_COMMAND = {SWAP_LR_COMMAND}")
    print("==============================================")

    warmup_models()

    prev_steering = 0.0
    mask_miss_count = 0
    circle_count = 0
    sign_counts = {
        label: 0
        for label in SIGN_LABEL_TO_RIGHT_BOX_COUNT
    }
    drive_mode = "ROAD_FOLLOW"
    force_right_start_time = None
    post_force_stop_start_time = None
    lane_diverge_count = 0
    lane_gap_state = make_lane_gap_state()

    stop_vehicle()
    time.sleep(1.0)

    try:
        while True:
            left_frame = left_camera.read()
            right_frame = right_camera.read()

            if left_frame is None and right_frame is None:
                print("[CAMERA] both frames missing, skip")
                continue

            if left_frame is None or right_frame is None:
                print(
                    f"[CAMERA] "
                    f"left={'OK' if left_frame is not None else 'MISS'} "
                    f"right={'OK' if right_frame is not None else 'MISS'}"
                )

            if drive_mode == "FORCE_RIGHT":
                elapsed_force_right = time.time() - force_right_start_time

                if elapsed_force_right >= FORCE_RIGHT_SECONDS:
                    print(
                        f"[FORCE RIGHT DONE] "
                        f"{elapsed_force_right:.2f}s elapsed. "
                        f"Stop before lane_v2 follow."
                    )
                    stop_vehicle()
                    post_force_stop_start_time = time.time()
                    drive_mode = "POST_FORCE_STOP"
                    continue

                print(
                    f"[FORCE RIGHT] "
                    f"steering={FORCE_RIGHT_STEERING:+.3f} "
                    f"speed={FORCE_RIGHT_SPEED:.2f} "
                    f"elapsed={elapsed_force_right:.2f}/"
                    f"{FORCE_RIGHT_SECONDS:.2f}s"
                )

                update_vehicle_motion(
                    FORCE_RIGHT_STEERING,
                    FORCE_RIGHT_SPEED
                )
                time.sleep(0.03)
                continue

            if drive_mode == "POST_FORCE_STOP":
                elapsed_post_stop = time.time() - post_force_stop_start_time

                if elapsed_post_stop >= POST_FORCE_STOP_SECONDS:
                    print(
                        f"[POST FORCE STOP DONE] "
                        f"{elapsed_post_stop:.2f}s elapsed. "
                        f"Switch to lane_v2 follow."
                    )
                    pid.reset()
                    prev_steering = 0.0
                    mask_miss_count = 0
                    lane_diverge_count = 0
                    lane_gap_state = make_lane_gap_state()
                    drive_mode = "LANE_V2_FOLLOW"
                    continue

                print(
                    f"[POST FORCE STOP] "
                    f"elapsed={elapsed_post_stop:.2f}/"
                    f"{POST_FORCE_STOP_SECONDS:.2f}s"
                )
                stop_vehicle()
                time.sleep(0.03)
                continue

            if drive_mode == "LANE_V2_FOLLOW":
                (
                    steering,
                    valid_lane,
                    lane_diverge_count,
                    lane_gap_state,
                    debug
                ) = get_steering_from_lane_v2_boxes(
                    left_frame,
                    right_frame,
                    prev_steering,
                    lane_diverge_count,
                    lane_gap_state
                )

                if valid_lane:
                    prev_steering = steering
                    mask_miss_count = 0

                    print(
                        f"[LANE_V2] "
                        f"camera={debug['camera']} "
                        f"boxes={debug['box_count']} "
                        f"mode={debug['target_mode']} "
                        f"target_conf={debug['target_conf']:.2f} "
                        f"gap={debug['pair_gap']} "
                        f"min_gap={debug['min_pair_gap']} "
                        f"threshold={debug['diverge_threshold']} "
                        f"trend={debug['gap_trend']} "
                        f"dec_streak={debug['decrease_streak']} "
                        f"inc_streak={debug['increase_streak']} "
                        f"diverge_active={debug['diverge_active']} "
                        f"diverge_count={debug['diverge_count']} "
                        f"error={debug['error_norm']:+.3f} "
                        f"raw={debug['raw_steering']:+.3f} "
                        f"steer={steering:+.3f}"
                    )

                else:
                    mask_miss_count += 1

                    if mask_miss_count <= MAX_MASK_HOLD_FRAMES:
                        steering = prev_steering
                        print(
                            f"[LANE_V2 MISS] hold prev steering "
                            f"{mask_miss_count}/{MAX_MASK_HOLD_FRAMES}"
                        )
                    else:
                        print("[LANE_V2 LOST] stop for safety")
                        stop_vehicle()
                        continue

                update_vehicle_motion(
                    steering,
                    BASE_SPEED
                )

                time.sleep(0.03)
                continue

            # =====================================
            # 1. Circle sign detection
            # =====================================
            detected_sign_labels = detect_sign_labels_dual(
                left_frame,
                right_frame
            )

            for label in detected_sign_labels:
                sign_counts[label] += 1

            if len(detected_sign_labels) > 0:
                print(
                    f"[SIGN DETECT] "
                    f"labels={sorted(detected_sign_labels)} "
                    f"counts={sign_counts}"
                )

            (
                circle_detected,
                circle_width,
                circle_conf,
                circle_source
            ) = detect_circle_dual(
                left_frame,
                right_frame
            )

            if circle_detected:
                circle_count += 1
            else:
                circle_count = 0

            if circle_count >= CIRCLE_CONFIRM_FRAMES:
                (
                    selected_right_box_count,
                    selected_sign_label,
                    selected_sign_count
                ) = choose_lane_v2_right_box_count(
                    sign_counts,
                    LANE_V2_RIGHT_BOX_COUNT
                )

                LANE_V2_RIGHT_BOX_COUNT = selected_right_box_count

                print(
                    f"[SIGN STOP] Circle detected for "
                    f"{circle_count} frame(s). "
                    f"source={circle_source} "
                    f"width={circle_width:.1f}px "
                    f"conf={circle_conf:.2f}. "
                    f"sign={selected_sign_label} "
                    f"sign_count={selected_sign_count} "
                    f"LANE_V2_RIGHT_BOX_COUNT={LANE_V2_RIGHT_BOX_COUNT}. "
                    f"Stop, then force right."
                )
                stop_vehicle()
                time.sleep(SIGN_STOP_SECONDS)
                pid.reset()
                prev_steering = 0.0
                mask_miss_count = 0
                drive_mode = "FORCE_RIGHT"
                force_right_start_time = time.time()
                continue

            # =====================================
            # 2. Dual segmentation + PID steering
            # =====================================
            steering, valid_mask, debug = get_steering_from_dual_segmentation(
                left_frame,
                right_frame,
                prev_steering
            )

            if valid_mask:
                prev_steering = steering
                mask_miss_count = 0

                print(
                    f"[PID] "
                    f"source={debug['source']} "
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
