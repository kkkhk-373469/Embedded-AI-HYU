from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera
import torch
import torchvision
from torchvision import transforms
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
# =====================================
# 차량 연결
# =====================================
base = BaseController('/dev/ttyUSB0', 115200)
# =====================================
# 카메라 연결
# =====================================
camera = CSICamera(
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)
# =====================================
# DEVICE
# =====================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
# =====================================
# 모델 생성
# =====================================
def get_model():
    model = torchvision.models.alexnet(
        num_classes=2
    )
    return model
# =====================================
# 모델 로드
# =====================================
def load_model(weight_path):
    model = get_model()
    model.load_state_dict(
        torch.load(
            weight_path,
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded: {weight_path}")
    return model
# =====================================
# 모델들
# =====================================
right_model = load_model(
    'right_3.pth'
)
right2_model = load_model(
    'road_right.pth'
)
# =====================================
# YOLO Segmentation 모델
# =====================================
start_left_seg_model = YOLO(
    'left_start.engine',
    task='segment'
)
normal_left_seg_model = YOLO(
    'left.engine',
    task='segment'
)
normal_right_seg_model = YOLO(
    'right.engine',
    task='segment'
)
rotation_seg_model = YOLO(
    'rotation11.engine',
    task='segment'
)
# =====================================
# YOLO
# =====================================
yolo_model = YOLO('best_final.pt')
circle_model = YOLO('best_circle.pt')
sinho_model = YOLO('best_final.pt')
LR_model = YOLO('best_LR_2.pt')
# =====================================
# 접근 차량 Rover 탐지 모델
# =====================================
# Rover가 접근했다가 멀어지는지 판단하는 접근 차량 탐지 모델.
# 회전 구간 조향용 rotation.pt와는 목적이 다르므로 기존 rover.pt를 그대로 사용한다.
rover_model = YOLO('rover.pt')
print("[INFO] Sign classes:", yolo_model.names)
print("[INFO] Circle classes:", circle_model.names)
print("[INFO] Rover classes:", rover_model.names)
# =====================================
# 이미지 전처리
# =====================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
def preprocess(frame):
    image = transform(frame)
    return image.unsqueeze(0).to(device)
# =====================================
# MAX
# =====================================
MAX_STEER = 0.8
MAX_SPEED = 0.5

# 두 경로 공통: 두 번째 회전교차로 통과 후,
# SECOND_NORMAL_RIGHT(right.engine)로 3초간 우회전한 뒤 강제 직진
SECOND_NORMAL_RIGHT_FORCE_STRAIGHT_DELAY = 5.0
# =====================================
# Circle 전용 탐지 조건
# =====================================
# best_circle.pt에서 class 이름이 circle 또는 Circle이어야 함
CIRCLE_CONF = 0.60
CIRCLE_MIN_WIDTH = 280
# =====================================
# Rover 통과 판단 조건
# =====================================
# confidence 0.70 이상으로 탐지된 Rover만 크기 비교에 사용
ROVER_CONF = 0.657

# bbox 면적 / 전체 프레임 면적. 너무 작은 검출은 사용하지 않음
ROVER_MIN_AREA_RATIO = 0.001

# 관측된 최대 bbox의 85% 이하가 되면 멀어지는 후보로 판단
ROVER_AWAY_RATIO = 0.80

# 멀어지는 상태가 3프레임 연속 확인되면 출발
ROVER_AWAY_CONFIRM_FRAMES = 3

# =====================================
# Segmentation 사용할 모드
# =====================================
SEGMENTATION_MODES = {
    "START_LEFT",           
    "NORMAL_LEFT",
    "NORMAL_RIGHT",
    "ROTATION_ONLY",
    "SECOND_ROTATION_ONLY",
    "SECOND_NORMAL_RIGHT"
}
# =====================================
# STAGE PARAMS
# =====================================
STAGE_PARAMS = {
    # =================================
    # 시작 좌회전
    # =================================
    "START_LEFT": {

        "model": start_left_seg_model,

        "speed": 0.25,

        "turn_strength": 0.27,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0.00,

        "Kp": 1.15,

        "use_deadzone": True,

        "deadzone_min": 0,

        "deadzone_max": 0.0,

        "L_offset": 0.0,

        "R_offset": 0.0
    },

    # =================================
    # 처음 RIGHT 선택 후 Rover가 멀어질 때까지 정지
    # =================================
    "WAIT_ROVER_CLEAR": {

        "model": None,

        "speed": 0.0,

        "turn_strength": 0.0,

        "use_fixed_steering": True,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0.0,

        "Kp": 0.0,

        "use_deadzone": False,

        "deadzone_min": 0.0,

        "deadzone_max": 0.0,

        "L_offset": 0.0,

        "R_offset": 0.0
    },

    # =================================
    # 처음 LEFT였고 Circle 감지 후 Rover가 멀어질 때까지 정지
    # =================================
    "WAIT_SECOND_LEFT_ROVER_CLEAR": {

        "model": None,

        "speed": 0.0,

        "turn_strength": 0.0,

        "use_fixed_steering": True,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0.0,

        "Kp": 0.0,

        "use_deadzone": False,

        "deadzone_min": 0.0,

        "deadzone_max": 0.0,

        "L_offset": 0.0,

        "R_offset": 0.0
    },

    # =================================
    # 첫번째 회전교차로 우회전 진입
    # =================================
    "ROUNDABOUT_RIGHT_ENTER": {

        "model": right2_model,

        "speed": 0.23,

        "turn_strength": 0.53,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 7.0,

        "bias": 0,

        "Kp": 2.5,

        "use_deadzone": True,

        "deadzone_min": 0.0,

        "deadzone_max": 0.08,

        "L_offset": 0.00,

        "R_offset": 0.0
    },

    # =================================
    # 첫번째 회전교차로 좌회전 진입
    # =================================
    "ROUNDABOUT_LEFT_ENTER": {

        "model": right2_model,

        "speed": 0.25,

        "turn_strength": 0.51,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 2.8,

        "bias": 0,

        "Kp": 2.5,

        "use_deadzone": True,

        "deadzone_min": 0.0,

        "deadzone_max": 0.08,

        "L_offset": 0.00,

        "R_offset": 0.0
    },

    # =================================
    # rotation only
    # =================================
    "ROTATION_ONLY": {

        "model": rotation_seg_model,

        "speed": 0.228,

        "turn_strength": 0.3,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 6.1,

        "bias": 0.0,

        "Kp": 1.4,

        "use_deadzone": False,

        "deadzone_min": 0.0,

        "deadzone_max": 0.0,

        "L_offset": 0.00,

        "R_offset": 0.00
    },

    # =================================
    # 일반 우회전
    # =================================
    "NORMAL_RIGHT": {

        "model": normal_right_seg_model,

        "speed": 0.22,

        "turn_strength": 0.254,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0,

        "Kp": 1.2,

        "use_deadzone": True,

        "deadzone_min": 0,

        "deadzone_max": 0.00,

        "L_offset": 0.0,

        "R_offset": 0.00
    },

    # =================================
    # 일반 좌회전
    # =================================
    "NORMAL_LEFT": {

        "model": start_left_seg_model, #normal_left_seg_model

        "speed": 0.25,

        "turn_strength": 0.305, #0.31

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0.0,

        "Kp": 1.08, #1.08

        "use_deadzone": True,

        "deadzone_min": 0,

        "deadzone_max": 0.0,

        "L_offset": 0.00,

        "R_offset": 0.0
    },

    # =================================
    # 두번째 회전교차로 LEFT 루트
    # =================================
    "SECOND_ROUNDABOUT_LEFT": {

        "model": right2_model,

        "speed": 0.25,

        "turn_strength": 0.5,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 9.0,

        "bias": 0,

        "Kp": 2.5,

        "use_deadzone": True,

        "deadzone_min": 0.0,

        "deadzone_max": 0.00,

        "L_offset": 0.00,

        "R_offset": 0.0
    },

    # =================================
    # 두번째 회전교차로 RIGHT 진입
    # =================================
    "SECOND_ROUNDABOUT_RIGHT_ENTER": {

        "model": right2_model,

        "speed": 0.18,

        "turn_strength": 0.5,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 5.2,

        "bias": 0,

        "Kp": 2.5,

        "use_deadzone": True,

        "deadzone_min": 0.0,

        "deadzone_max": 0.08,

        "L_offset": 0.00,

        "R_offset": 0.0
    },

    # =================================
    # 두번째 rotation only
    # =================================
    "SECOND_ROTATION_ONLY": {

        "model": rotation_seg_model,

        "speed": 0.23,

        "turn_strength": 0.32,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 6.1,

        "bias": 0.0,

        "Kp": 1.0,

        "use_deadzone": False,

        "deadzone_min": 0.0,

        "deadzone_max": 0.0,

        "L_offset": 0.00,

        "R_offset": 0.00
    },

    # =================================
    # 두번째 회전교차로 이후 일반 우회전
    # =================================
    "SECOND_NORMAL_RIGHT": {

        # 두 번째 회전교차로 통과 후 일반 우회전: right.engine segmentation 사용
        "model": normal_right_seg_model,

        "speed": 0.22,

        "turn_strength": 0.254,

        "use_fixed_steering": False,

        "fixed_steering": 0.0,

        "duration": 9999,

        "bias": 0,

        "Kp": 1.1,

        "use_deadzone": True,

        "deadzone_min": 0,

        "deadzone_max": 0.0,

        "L_offset": 0.08,

        "R_offset": 0.00
    }
}

# =================================
# SIGN DETECTION CONFIG
# =================================
SIGN_CONFIG = {

    "LEFT": {

        "conf": 0.5,

        "min_width": 15
    },

    "RIGHT": {

        "conf": 0.5
        ,

        "min_width": 15
    },

    "SINHO": {

        "conf": 0.40,

        "min_width": 100
    },

    "SLOW": {

        "conf": 0.6,

        "min_width": 100
    },

    "STOP": {

        "conf": 0.5,

        "min_width": 100
    }
}

# =====================================
# 시작 설정
# =====================================
current_stage = 0
drive_mode = "START_LEFT"
stage_start_time = time.time()
roundabout_direction = None

# =====================================
# 시작 표지판 카운트
# =====================================
left_sign_count = 0
right_sign_count = 0
circle_detect_count = 0
second_circle_detect_count = 0

# =====================================
# 표지판 탐지 카운트
# =====================================
left_detect_count = 0
right_detect_count = 0

# =====================================
# 이벤트 상태
# =====================================
event_mode = None
event_start_time = 0.0

# =====================================
# 이벤트 누적 시간
# =====================================
paused_time_total = 0.0
pause_start_time = 0.0

# =====================================
# 이벤트 쿨타임
# =====================================
event_cooldown = 10.0
last_event_time = -999

# =================================
# 이벤트 실행 횟수
# =================================
sinho_count = 0
stop_count = 0
slow_count = 0

# =====================================
# SINHO release
# =====================================
sinho_miss_count = 0
SINHO_MISS_THRESHOLD = 20

# =====================================
# Steering smoothing / mask 미탐지시 이전 steering 유지
# =====================================
prev_steering = 0.0

# 기본 segmentation 조향 smoothing
STEERING_SMOOTHING = 0.44

# 회전교차로 내부 rotation 구간 전용 smoothing
# 값이 클수록 이전 steering을 더 많이 반영해 조향 변화가 부드러워진다.
ROTATION_STEERING_SMOOTHING = 0.67

# =====================================
# Rover bbox 추적 / Rover 대기 시간 상태
# =====================================
rover_seen = False
rover_max_area_ratio = 0.0
rover_away_count = 0

# Rover 때문에 정지하기 시작한 시각.
# 기존 event의 pause_start_time과 분리하여 충돌을 방지.
rover_pause_start_time = 0.0

# =====================================
# TensorRT Segmentation Engine Warm-up
# =====================================
# TensorRT engine은 해당 모델의 첫 predict() 시점에 실제 로딩/초기화가 발생할 수 있다.
# 주행 도중 모드가 바뀌는 순간 로딩 지연이 생기지 않도록 출발 전에 한 번씩 예열한다.
def warmup_segmentation_engines():
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    engine_models = [
        ("left_start.engine", start_left_seg_model),
        ("left.engine", normal_left_seg_model),
        ("right.engine", normal_right_seg_model),
        ("rotation11.engine", rotation_seg_model)
    ]

    print("[WARMUP] TensorRT segmentation engines loading...")

    for model_name, model in engine_models:
        model.predict(
            source=dummy_frame,
            imgsz=640,
            verbose=False
        )
        print(f"[WARMUP] Ready: {model_name}")

    print("[WARMUP] All segmentation engines ready.")


warmup_segmentation_engines()

print(
    f"[START] "
    f"Stage: {current_stage} "
    f"| Mode: {drive_mode} "
    f"| Direction: {roundabout_direction}"
)

# =====================================
# 비동기 제어
# =====================================
def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({
            'T': 1,
            'L': L,
            'R': R
        })
    threading.Thread(
        target=worker,
        daemon=True
    ).start()

# =====================================
# clip
# =====================================
def clip(val, max_val):
    return max(
        min(val, max_val),
        -max_val
    )

# =====================================
# 차량 제어
# =====================================
def update_vehicle_motion(
    steering,
    speed,
    turn_strength,
    L_offset,
    R_offset
):
    steering = clip(
        steering,
        MAX_STEER
    )

    L = speed - steering * turn_strength
    R = speed + steering * turn_strength

    L += L_offset
    R += R_offset

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    send_control_async(R, L)

    print(
        f"[AUTO] "
        f"Mode: {drive_mode:<35} "
        f"Steer: {steering:+.2f} "
        f"L: {L:+.2f} "
        f"R: {R:+.2f}"
    )

# =====================================
# LEFT / RIGHT 방향 표지판 탐지
# START_LEFT 구간에서만 사용
# =====================================
def detect_left_right(frame):

    lr_results = LR_model.predict(
        source=frame,
        imgsz=640,
        verbose=False
    )

    for r in lr_results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            label = LR_model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]
            box_width = x2 - x1

            if (
                label == "Left"
                and box_width >= SIGN_CONFIG["LEFT"]["min_width"]
                and conf >= SIGN_CONFIG["LEFT"]["conf"]
            ):
                return "LEFT"

            elif (
                label == "Right"
                and box_width >= SIGN_CONFIG["RIGHT"]["min_width"]
                and conf >= SIGN_CONFIG["RIGHT"]["conf"]
            ):
                return "RIGHT"

    return None


# =====================================
# STOP / SLOW / SINHO 이벤트 탐지
# 일반 주행 구간에서만 사용
# =====================================
def detect_event_sign(frame):

    event_results = yolo_model.predict(
        source=frame,
        imgsz=640,
        verbose=False
    )

    sinho_results = sinho_model.predict(
        source=frame,
        imgsz=640,
        verbose=False
    )

    for r in event_results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]
            box_width = x2 - x1

            if (
                label == "Slow"
                and box_width >= SIGN_CONFIG["SLOW"]["min_width"]
                and conf >= SIGN_CONFIG["SLOW"]["conf"]
            ):
                return "SLOW"

            elif (
                label == "Stop"
                and box_width >= SIGN_CONFIG["STOP"]["min_width"]
                and conf >= SIGN_CONFIG["STOP"]["conf"]
            ):
                return "STOP"

    for r in sinho_results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            label = sinho_model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]
            box_height = y2 - y1

            if (
                label == "Sinho"
                and box_height >= SIGN_CONFIG["SINHO"]["min_width"]
                and conf >= SIGN_CONFIG["SINHO"]["conf"]
            ):
                return "SINHO"

    return None

# =====================================
# Circle 전용 탐지
# =====================================
def detect_circle(frame):
    """
    best_circle.pt를 사용해 두 번째 회전교차로 진입 기준인 Circle만 탐지한다.
    Circle confidence와 bbox width가 기준 이상일 때만 True를 반환한다.
    """
    results = circle_model.predict(
        source=frame,
        conf=CIRCLE_CONF,
        imgsz=640,
        verbose=False
    )

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

            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            box_width = x2 - x1

            if box_width >= CIRCLE_MIN_WIDTH:
                print(
                    f"[CIRCLE] DETECTED conf={conf:.2f} "
                    f"width={box_width:.1f}"
                )
                return True

    return False

# =====================================
# Rover bbox 크기 기반 출발 판단
# =====================================
def reset_rover_tracking():

    global rover_seen
    global rover_max_area_ratio
    global rover_away_count

    rover_seen = False
    rover_max_area_ratio = 0.0
    rover_away_count = 0


def detect_rover_area(frame):

    results = rover_model.predict(
        source=frame,
        conf=ROVER_CONF,
        imgsz=640,
        verbose=False
    )

    height, width = frame.shape[:2]
    frame_area = height * width

    largest_area_ratio = 0.0
    largest_conf = 0.0

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                box.xyxy[0].detach().cpu().numpy()
            )

            area_ratio = float(
                ((x2 - x1) * (y2 - y1))
                / frame_area
            )

            if area_ratio > largest_area_ratio:
                largest_area_ratio = area_ratio
                largest_conf = conf

    return largest_area_ratio, largest_conf


def check_rover_clear(frame):

    global rover_seen
    global rover_max_area_ratio
    global rover_away_count

    area_ratio, conf = detect_rover_area(frame)

    # conf >= ROVER_CONF인 유효 Rover가 없으면 출발하지 않음
    if area_ratio < ROVER_MIN_AREA_RATIO:
        print("[ROVER] WAIT - no valid Rover detection")
        return False

    rover_seen = True

    # 더 커지고 있으면 아직 접근 중
    if area_ratio > rover_max_area_ratio:
        rover_max_area_ratio = area_ratio
        rover_away_count = 0

        print(
            f"[ROVER] APPROACHING/CLOSE "
            f"conf={conf:.2f} area={area_ratio:.4f} "
            f"max={rover_max_area_ratio:.4f}"
        )
        return False

    # 최대 크기보다 충분히 작아진 유효 검출이 연속되면 출발
    if area_ratio <= rover_max_area_ratio * ROVER_AWAY_RATIO:
        rover_away_count += 1

        print(
            f"[ROVER] MOVING AWAY "
            f"conf={conf:.2f} area={area_ratio:.4f} "
            f"max={rover_max_area_ratio:.4f} "
            f"count={rover_away_count}/{ROVER_AWAY_CONFIRM_FRAMES}"
        )

        if rover_away_count >= ROVER_AWAY_CONFIRM_FRAMES:
            print("[ROVER] CLEAR - moving away -> START")
            return True
    else:
        rover_away_count = 0
        print(
            f"[ROVER] HOLD "
            f"conf={conf:.2f} area={area_ratio:.4f} "
            f"max={rover_max_area_ratio:.4f}"
        )

    return False

# =====================================
# STAGE UPDATE
# =====================================
def update_stage(
    now,
    detected_sign,
    circle_detected,
    rover_clear
):

    global current_stage
    global stage_start_time
    global drive_mode
    global roundabout_direction
    global paused_time_total
    global rover_pause_start_time

    elapsed = now - stage_start_time

    # =================================
    # STAGE 0
    # 회전교차로 진입 전 Left / Right 표지판 카운트
    # =================================
    if current_stage == 0:

        global left_sign_count
        global right_sign_count
        global circle_detect_count

        elapsed = time.time() - stage_start_time

        if detected_sign == "LEFT":
            left_sign_count += 1
            print(f"[LEFT COUNT] {left_sign_count}")

        elif detected_sign == "RIGHT":
            right_sign_count += 1
            print(f"[RIGHT COUNT] {right_sign_count}")

        # =================================
        # Circle detect count
        # =================================
        if circle_detected:

            circle_detect_count += 1

            print(
                f"[CIRCLE COUNT] "
                f"{circle_detect_count}"
            )

        else:

            circle_detect_count = 0

        # =================================
        # Circle 연속 탐지 성공
        # =================================
        if circle_detect_count >= 3:

            print(
                f"[FINAL COUNT] "
                f"L={left_sign_count} "
                f"R={right_sign_count}"
            )

            # =============================
            # RIGHT 우세
            # =============================
            if right_sign_count > left_sign_count:

                roundabout_direction = "RIGHT"

                current_stage = 10

                drive_mode = "WAIT_ROVER_CLEAR"

                stage_start_time = time.time()

                reset_rover_tracking()

                rover_pause_start_time = time.time()

                base.base_json_ctrl({

                    'T': 1,

                    'L': 0.0,

                    'R': 0.0
                })

                print(
                    "[STAGE] RIGHT SELECTED "
                    "-> WAIT_ROVER_CLEAR"
                )

            # =============================
            # LEFT 우세
            # =============================
            else:

                roundabout_direction = "LEFT"

                current_stage = 22

                drive_mode = (
                    "ROUNDABOUT_LEFT_ENTER"
                )

                stage_start_time = time.time()

                print(
                    "[STAGE] LEFT SELECTED "
                    "-> ROUNDABOUT_LEFT_ENTER"
                )

    # =================================
    # STAGE 10
    # 처음 RIGHT 경로: Rover 통과 대기
    # =================================
    elif current_stage == 10:

        if rover_clear:

            rover_wait_duration = time.time() - rover_pause_start_time
            paused_time_total += rover_wait_duration

            print(
                f"[PAUSE] Rover wait added: "
                f"{rover_wait_duration:.2f}s "
                f"| paused_time_total={paused_time_total:.2f}s"
            )

            current_stage = 21
            stage_start_time = time.time()
            drive_mode = "ROUNDABOUT_RIGHT_ENTER"

            print(
                "[STAGE] FIRST RIGHT ROVER CLEAR "
                "-> ROUNDABOUT_RIGHT_ENTER"
            )

    elif current_stage == 21:

        elapsed = time.time() - stage_start_time

        if elapsed > STAGE_PARAMS[
            "ROUNDABOUT_RIGHT_ENTER"
        ]["duration"]:

            current_stage = 3

            stage_start_time = time.time()

            paused_time_total = 0.0

            drive_mode = "NORMAL_LEFT"

    elif current_stage == 22:

        elapsed = time.time() - stage_start_time

        if drive_mode == "ROUNDABOUT_LEFT_ENTER":

            if elapsed > STAGE_PARAMS[
                "ROUNDABOUT_LEFT_ENTER"
            ]["duration"]:

                stage_start_time = time.time()

                drive_mode = "ROTATION_ONLY"

        elif drive_mode == "ROTATION_ONLY":

            if elapsed > STAGE_PARAMS[
                "ROTATION_ONLY"
            ]["duration"]:

                current_stage = 3

                stage_start_time = time.time()

                paused_time_total = 0.0

                drive_mode = "NORMAL_RIGHT"

                base.base_json_ctrl({

                    'T': 1,

                    'L': 0.0,

                    'R': 0.0
                })

    # =================================
    # STAGE 3: 일반 주행 중 두 번째 회전교차로 Circle 대기
    # =================================
    elif current_stage == 3:

        global second_circle_detect_count

        # =================================
        # Second circle detect count
        # =================================
        if circle_detected:

            second_circle_detect_count += 1

            print(
                f"[SECOND CIRCLE COUNT] "
                f"{second_circle_detect_count}"
            )

        else:

            second_circle_detect_count = 0

        # =================================
        # Circle 연속 탐지 성공
        # =================================
        if second_circle_detect_count >= 2:

            # =============================
            # 처음 LEFT
            # =============================
            if roundabout_direction == "LEFT":

                current_stage = 40

                stage_start_time = time.time()

                drive_mode = (
                    "WAIT_SECOND_LEFT_ROVER_CLEAR"
                )

                reset_rover_tracking()

                rover_pause_start_time = time.time()

                base.base_json_ctrl({

                    'T': 1,

                    'L': 0.0,

                    'R': 0.0
                })

                print(
                    "[STAGE] "
                    "CIRCLE DETECTED AFTER FIRST LEFT "
                    "-> WAIT_SECOND_LEFT_ROVER_CLEAR"
                )

            # =============================
            # 처음 RIGHT
            # =============================
            elif roundabout_direction == "RIGHT":

                current_stage = 42

                stage_start_time = time.time()

                paused_time_total = 0.0

                drive_mode = (
                    "SECOND_ROUNDABOUT_RIGHT_ENTER"
                )

                print(
                    "[STAGE] "
                    "CIRCLE DETECTED AFTER FIRST RIGHT "
                    "-> SECOND_ROUNDABOUT_RIGHT_ENTER"
                )

    # =================================
    # STAGE 40
    # 처음 LEFT였을 때 두 번째 회전교차로 진입 전 Rover 대기
    # =================================
    elif current_stage == 40:

        if rover_clear:

            rover_wait_duration = time.time() - rover_pause_start_time
            paused_time_total += rover_wait_duration

            print(
                f"[PAUSE] Rover wait added: "
                f"{rover_wait_duration:.2f}s "
                f"| paused_time_total={paused_time_total:.2f}s"
            )

            current_stage = 41
            stage_start_time = time.time()
            drive_mode = "SECOND_ROUNDABOUT_LEFT"

            print(
                "[STAGE] SECOND LEFT ROVER CLEAR "
                "-> SECOND_ROUNDABOUT_LEFT"
            )

    elif current_stage == 41:
        elapsed = time.time() - stage_start_time

        if elapsed > STAGE_PARAMS[
            "SECOND_ROUNDABOUT_LEFT"
        ]["duration"]:

            # 처음 LEFT 경로도 두 번째 회전교차로 이후 일반 우회전으로 진입
            current_stage = 43
            stage_start_time = time.time()
            paused_time_total = 0.0
            drive_mode = "SECOND_NORMAL_RIGHT"

            print(
                "[STAGE] FIRST LEFT ROUTE: "
                "SECOND_ROUNDABOUT_LEFT -> SECOND_NORMAL_RIGHT "
                "(FORCE STRAIGHT AFTER 3.0s)"
            )

    elif current_stage == 42:
        elapsed = time.time() - stage_start_time

        if (

            drive_mode
            == "SECOND_ROUNDABOUT_RIGHT_ENTER"
        ):

            if elapsed > STAGE_PARAMS[
                "SECOND_ROUNDABOUT_RIGHT_ENTER"
            ]["duration"]:

                stage_start_time = time.time()
                paused_time_total = 0.0

                drive_mode = (
                    "SECOND_ROTATION_ONLY"
                )

        elif drive_mode == "SECOND_ROTATION_ONLY":

            if elapsed > STAGE_PARAMS[
                "SECOND_ROTATION_ONLY"
            ]["duration"]:
                paused_time_total = 0.0
                drive_mode = (
                    "SECOND_NORMAL_RIGHT"
                )

                stage_start_time = time.time()

                print(
                    "[STAGE] "
                    "SECOND_ROTATION_ONLY "
                    "-> SECOND_NORMAL_RIGHT"
                )

# =====================================
# MAIN LOOP
# =====================================
try:

    while True:

        frame = camera.read()

        height, width = frame.shape[:2]

        center_x = width / 2

        now = time.time()

        # =================================
        # 현재 모드에 필요한 탐지 모델만 실행
        # =================================
        detected_sign = None
        circle_detected = False
        rover_clear = False

        if drive_mode == "START_LEFT":
            # 방향표지판으로 LEFT/RIGHT 경로 결정 + 첫 Circle 도착 확인
            detected_sign = detect_left_right(frame)
            circle_detected = detect_circle(frame)

        elif drive_mode in (
            "NORMAL_LEFT",
            "NORMAL_RIGHT"
        ):
            # 이벤트 탐지 + 두 번째 Circle 도착 확인
            detected_sign = detect_event_sign(frame)
            circle_detected = detect_circle(frame)

        elif drive_mode == "SECOND_NORMAL_RIGHT":
            # 두 번째 회전교차로 이후에는 이벤트만 확인
            detected_sign = detect_event_sign(frame)

        elif drive_mode in (
            "WAIT_ROVER_CLEAR",
            "WAIT_SECOND_LEFT_ROVER_CLEAR"
        ):
            # 정지 대기 구간에서는 Rover만 확인
            rover_clear = check_rover_clear(frame)

        # ROUNDABOUT_* / ROTATION_ONLY / SECOND_ROTATION_ONLY는
        # 추가 탐지를 하지 않고 조향 모델 하나만 실행

        update_stage(
            now,
            detected_sign,
            circle_detected,
            rover_clear
        )

        # =================================
        # 일반 주행 중 이벤트 감지
        # =================================
        if (

            drive_mode == "NORMAL_LEFT"

            or

            drive_mode == "NORMAL_RIGHT"

            or

            drive_mode == "SECOND_NORMAL_RIGHT"
        ):

            # =================================
            # 이벤트 쿨타임
            # =================================
            if (

                event_mode is None

                and

                now - last_event_time
                > event_cooldown
            ):

                # =============================
                # SINHO
                # =============================
                if detected_sign == "SINHO" and sinho_count < 3:

                    event_mode = "SINHO"

                    sinho_miss_count = 0

                    event_start_time = now

                    pause_start_time = now

                    last_event_time = now

                    sinho_count += 1

                    print("[EVENT] SINHO")

                # =============================
                # STOP
                # =============================
                elif detected_sign == "STOP" and stop_count < 2:

                    event_mode = "STOP"

                    event_start_time = now

                    pause_start_time = now

                    last_event_time = now

                    stop_count += 1

                    print("[EVENT] STOP")

                # =============================
                # SLOW
                # =============================
                elif detected_sign == "SLOW" and slow_count < 2:

                    event_mode = "SLOW"

                    event_start_time = now

                    pause_start_time = now

                    last_event_time = now

                    slow_count += 1

                    print("[EVENT] SLOW")

        params = STAGE_PARAMS[drive_mode]

        current_speed = params["speed"]
        motion_L_offset = params["L_offset"]
        motion_R_offset = params["R_offset"]

        # 두 경로 공통으로 두 번째 회전교차로 이후:
        # 3초 동안은 right.engine 기반 SECOND_NORMAL_RIGHT 조향을 수행하고,
        # 이후에는 조향 추론 없이 좌우 동일 출력으로 강제 직진한다.
        force_straight_after_second_right = (
            drive_mode == "SECOND_NORMAL_RIGHT"
            and (now - stage_start_time) >= SECOND_NORMAL_RIGHT_FORCE_STRAIGHT_DELAY
        )

        # =================================
        # FORCED STRAIGHT AFTER SECOND ROUNDABOUT
        # =================================
        if force_straight_after_second_right:

            steering = 0.0

            # SECOND_NORMAL_RIGHT의 기존 L_offset=0.08을 유지하면
            # steering=0이어도 좌우 출력이 달라지므로 직진 시에는 제거한다.
            motion_L_offset = 0.0
            motion_R_offset = 0.0

            print(
                f"[FORCE STRAIGHT] Route={roundabout_direction} "
                f"| elapsed={now - stage_start_time:.2f}s "
                f"| speed={current_speed:.2f}"
            )

        # =================================
        # FIXED STEERING
        # =================================
        elif params["use_fixed_steering"]:

            steering = params[
                "fixed_steering"
            ]

        else:

            # =================================
            # Segmentation 모드
            # =================================
            if drive_mode in SEGMENTATION_MODES:

                # =============================
                # 현재 segmentation 모델 선택
                # =============================
                # START_LEFT / NORMAL_LEFT는 left.pt를 사용하고,
                # NORMAL_RIGHT는 right.pt를 사용하며,
                # ROTATION_ONLY / SECOND_ROTATION_ONLY는 rotation.pt를 사용한다.
                seg_model = params["model"]

                results = seg_model.predict(
                    source=frame,
                    imgsz=640,
                    verbose=False
                )

                # mask가 일시적으로 미탐지되면 0으로 급변시키지 않고
                # 직전 조향값을 유지하여 조향 튐을 줄인다.
                steering = prev_steering

                if len(results) > 0:

                    r = results[0]

                    # masks 객체가 있어도 실제 검출 mask 개수가 0일 수 있으므로
                    # 빈 mask일 때 masks[0]에 접근하지 않는다.
                    if (
                        r.masks is not None
                        and r.masks.data is not None
                        and r.masks.data.shape[0] > 0
                    ):

                        masks = (
                            r.masks.data
                            .cpu()
                            .numpy()
                        )

                        mask = masks[0]

                        mask = (
                            mask > 0.5
                        ).astype(np.uint8)

                        # =================================
                        # morphology
                        # =================================
                        kernel = np.ones(
                            (5,5),
                            np.uint8
                        )

                        # =============================
                        # 작은 노이즈 제거
                        # =============================
                        mask = cv2.morphologyEx(

                            mask,

                            cv2.MORPH_OPEN,

                            kernel
                        )

                        mask_h, mask_w = mask.shape

                        roi = mask[
                            int(mask_h * 0.7):,
                            :
                        ]

                        ys, xs = np.where(
                            roi > 0
                        )

                        if len(xs) > 0:

                            lane_center = np.mean(xs)

                            error = (
                                lane_center
                                - (mask_w / 2)
                            )

                            error_norm = (
                                error / (mask_w / 2)
                            )

                            steering = (

                                params["Kp"]
                                * error_norm

                                + params["bias"]
                            )

                            # =================================
                            # steering smoothing
                            # =================================
                            # ROTATION_ONLY / SECOND_ROTATION_ONLY에서만
                            # smoothing을 0.55로 높여 회전 중 조향 튐을 줄인다.
                            if drive_mode in (
                                "ROTATION_ONLY",
                                "SECOND_ROTATION_ONLY"
                            ):
                                smoothing = ROTATION_STEERING_SMOOTHING
                            else:
                                smoothing = STEERING_SMOOTHING

                            steering = (

                                smoothing
                                * prev_steering

                                +

                                (1 - smoothing)
                                * steering
                            )
                            if drive_mode == "NORMAL_LEFT":

                                MAX_STEER = 0.74

                                steering = np.clip(

                                    steering,

                                    -MAX_STEER,

                                    MAX_STEER
                                )

                            prev_steering = steering

                            
            # =================================
            # 기존 regression 모드
            # 회전교차로 진입 / SECOND_NORMAL_RIGHT만 해당
            # START_LEFT는 이제 left.pt segmentation 분기에서 처리됨
            # =================================
            else:

                input_tensor = preprocess(frame)

                with torch.no_grad():

                    output = params["model"](
                        input_tensor
                    )

                x, y = output[
                    0
                ].detach().cpu().numpy()

                pred_x = (
                    (x / 2 + 0.5)
                    * width
                )

                error = pred_x - center_x

                error_norm = error / center_x

                steering = (

                    params["Kp"]
                    * error_norm

                    + params["bias"]
                )

            # =================================
            # deadzone
            # =================================
            if params["use_deadzone"]:

                if (

                    params["deadzone_min"]

                    <= steering

                    <= params["deadzone_max"]
                ):

                    steering = 0.0

        steering = clip(
            steering,
            MAX_STEER
        )

        # =================================
        # EVENT OVERRIDE
        # =================================
        if event_mode == "SINHO":

            # =============================
            # 아직 신호등 보임
            # =============================
            if detected_sign == "SINHO":

                sinho_miss_count = 0

                print("[SINHO] RED")

            # =============================
            # 신호등 안 보임
            # =============================
            else:

                sinho_miss_count += 1

                print(
                    f"[SINHO] MISS "
                    f"{sinho_miss_count}"
                )

            # =============================
            # 10프레임 이상 미감지
            # =============================
            if (
                sinho_miss_count
                >= SINHO_MISS_THRESHOLD
            ):

                paused_time_total += (
                    now - pause_start_time
                )

                sinho_miss_count = 0

                event_mode = None

                print("[SINHO] GO")

            else:

                steering = 0.0
                current_speed = 0.0

        elif event_mode == "STOP":

            if now - event_start_time < 4.0:

                steering = 0.0
                current_speed = 0.0

            else:
                paused_time_total += (
                    now - pause_start_time
                )
                event_mode = None

        elif event_mode == "SLOW":

            if now - event_start_time < 4.0:

                current_speed *= 0.8

            else:
                paused_time_total += (
                    now - pause_start_time
                )
                event_mode = None

        # =================================
        # 차량 제어
        # =================================
        update_vehicle_motion(
            steering,
            current_speed,
            params["turn_strength"],
            motion_L_offset,
            motion_R_offset
        )
        time.sleep(0.03)

except KeyboardInterrupt:
    print("\nSTOP")

finally:
    base.base_json_ctrl({
        'T': 1,
        'L': 0.0,
        'R': 0.0
    })
    cv2.destroyAllWindows()
