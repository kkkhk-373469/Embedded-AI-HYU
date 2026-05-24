from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera

import torch
import torchvision
from torchvision import transforms

import cv2
import time
import threading

from ultralytics import YOLO


# =====================================
# 차량 / 카메라 연결
# =====================================
base = BaseController('/dev/ttyUSB0', 115200)

camera = CSICamera(
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)


# =====================================
# DEVICE / Lane Tracking 모델 로드
# =====================================
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


def get_model():
    return torchvision.models.alexnet(num_classes=2)


def load_model(weight_path):
    model = get_model()
    model.load_state_dict(
        torch.load(weight_path, map_location=device)
    )
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded lane model: {weight_path}")
    return model


left_model = load_model('left_4.pth')
right_model = load_model('right_3.pth')
rotation_only_model = load_model('road_rotation_only_model.pth')


# =====================================
# YOLO 모델
# =====================================
# 방향 표지판 / 신호 / Slow / Stop 탐지 모델
# Circle은 이 모델에서 탐지하지 않음
yolo_model = YOLO('best_real.pt')

# 두 번째 회전교차로 진입 기준인 Circle 전용 탐지 모델
circle_model = YOLO('best_circle.pt')

# 출발 판단에 사용할 Rover 탐지 모델
rover_model = YOLO('rover.pt')

print("[INFO] Sign/Event classes:", yolo_model.names)
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
# 공통 제한값
# =====================================
MAX_STEER = 2.0
MAX_SPEED = 0.5


# =====================================
# 표지판 인식 조건
# =====================================
SIGN_CONFIG = {
    "LEFT": {
        "conf": 0.80,
        "min_width": 40
    },
    "RIGHT": {
        "conf": 0.70,
        "min_width": 30
    },
    "SINHO": {
        "conf": 0.60,
        "min_height": 100
    },
    "SLOW": {
        "conf": 0.65,
        "min_width": 100
    },
    "STOP": {
        "conf": 0.65,
        "min_width": 110
    }
}

# =====================================
# Circle 전용 탐지 조건
# =====================================
# best_circle.pt에서 class 이름이 circle 또는 Circle이어야 함
CIRCLE_CONF = 0.70
CIRCLE_MIN_WIDTH = 300


# =====================================
# Rover 통과 판단 조건
# =====================================
ROVER_CONF = 0.70  # confidence 0.70 이상인 Rover bbox만 거리 판단에 사용

# bbox 면적 / 전체 프레임 면적. 이것보다 작으면 유효한 Rover로 보지 않음.
ROVER_MIN_AREA_RATIO = 0.001

# 대기 중 가장 컸던 bbox의 85% 이하로 작아지면 멀어지는 후보로 판단
ROVER_AWAY_RATIO = 0.85

# bbox가 작아지는 상태가 3프레임 연속 확인되면 출발
ROVER_AWAY_CONFIRM_FRAMES = 3

# Rover를 본 뒤 화면 밖으로 사라진 경우, 3프레임 연속 미검출이면 출발
ROVER_MISSING_CONFIRM_FRAMES = 3


# =====================================
# STAGE PARAMS
# =====================================
STAGE_PARAMS = {
    # =================================
    # 시작 주행: 이 구간에서 Left/Right 탐지 횟수 비교
    # =================================
    "START_LEFT": {
        "model": left_model,
        "speed": 0.18,
        "turn_strength": 0.76,
        "use_fixed_steering": False,
        "fixed_steering": 0.0,
        "duration": 10.0,
        "bias": +0.34,
        "Kp": 2.6,
        "use_deadzone": True,
        "deadzone_min": -0.02,
        "deadzone_max": 1.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # RIGHT로 판단된 경우 Rover가 멀어질 때까지 정지
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
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 처음 LEFT 경로였고 Circle을 본 뒤, Rover가 지나갈 때까지 정지
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
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 첫번째 회전교차로 RIGHT 표지판 루트
    # =================================
    "ROUNDABOUT_RIGHT_ENTER": {
        "model": None,
        "speed": 0.18,
        "turn_strength": 0.32,
        "use_fixed_steering": True,
        "fixed_steering": +0.30,
        "duration": 10.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 첫번째 회전교차로 LEFT 표지판 루트: 진입
    # =================================
    "ROUNDABOUT_LEFT_ENTER": {
        "model": None,
        "speed": 0.20,
        "turn_strength": 0.32,
        "use_fixed_steering": True,
        "fixed_steering": +0.30,
        "duration": 4.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 첫번째 회전교차로 LEFT 루트: 회전 유지
    # =================================
    "ROTATION_ONLY": {
        "model": rotation_only_model,
        "speed": 0.15,
        "turn_strength": 0.30,
        "use_fixed_steering": False,
        "fixed_steering": 0.0,
        "duration": 11.5,
        "bias": 0.15,
        "Kp": 2.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 일반 우회전 경로
    # =================================
    "NORMAL_RIGHT": {
        "model": right_model,
        "speed": 0.18,
        "turn_strength": 0.98,
        "use_fixed_steering": False,
        "fixed_steering": 0.0,
        "duration": 9999,
        "bias": -0.31,
        "Kp": 2.7,
        "use_deadzone": True,
        "deadzone_min": -1.0,
        "deadzone_max": 0.08,
        "L_offset": 0.02,
        "R_offset": 0.00
    },

    # =================================
    # 일반 좌회전 경로
    # =================================
    "NORMAL_LEFT": {
        "model": left_model,
        "speed": 0.18,
        "turn_strength": 0.98,
        "use_fixed_steering": False,
        "fixed_steering": 0.0,
        "duration": 9999,
        "bias": +0.34,
        "Kp": 2.5,
        "use_deadzone": True,
        "deadzone_min": -0.02,
        "deadzone_max": 1.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 두번째 회전교차로 LEFT 루트
    # =================================
    "SECOND_ROUNDABOUT_LEFT": {
        "model": None,
        "speed": 0.20,
        "turn_strength": 0.32,
        "use_fixed_steering": True,
        "fixed_steering": +0.30,
        "duration": 4.0,  # Rover 통과 확인 후 강제 조향 4초
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 두 번째 LEFT 루트: 강제 조향 이후 직진 10초
    # =================================
    "SECOND_LEFT_STRAIGHT": {
        "model": None,
        "speed": 0.18,
        "turn_strength": 0.0,
        "use_fixed_steering": True,
        "fixed_steering": 0.0,
        "duration": 10.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 미션 종료 후 정지 유지
    # =================================
    "MISSION_STOP": {
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
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 두번째 회전교차로 RIGHT 진입
    # =================================
    "SECOND_ROUNDABOUT_RIGHT_ENTER": {
        "model": None,
        "speed": 0.18,
        "turn_strength": 0.32,
        "use_fixed_steering": True,
        "fixed_steering": +0.30,
        "duration": 4.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 두번째 회전교차로 RIGHT 루트: 회전 유지
    # =================================
    "SECOND_ROTATION_ONLY": {
        "model": rotation_only_model,
        "speed": 0.15,
        "turn_strength": 0.30,
        "use_fixed_steering": False,
        "fixed_steering": 0.0,
        "duration": 10.0,  # 처음 RIGHT였던 경우, 두 번째 회전교차로 내부 회전 10초
        "bias": 0.15,
        "Kp": 2.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 처음 RIGHT 경로: 두 번째 회전교차로 회전 이후 오른쪽 강제 조향 3초
    # =================================
    "SECOND_RIGHT_FIXED_TURN": {
        "model": None,
        "speed": 0.18,
        "turn_strength": 0.32,
        "use_fixed_steering": True,
        "fixed_steering": +0.30,
        "duration": 3.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    },

    # =================================
    # 처음 RIGHT 경로: right_3.pth 이후 직진 명령 10초
    # =================================
    "SECOND_RIGHT_STRAIGHT": {
        "model": None,
        "speed": 0.18,
        "turn_strength": 0.0,
        "use_fixed_steering": True,
        "fixed_steering": 0.0,
        "duration": 10.0,
        "bias": 0.0,
        "Kp": 0.0,
        "use_deadzone": False,
        "deadzone_min": 0.0,
        "deadzone_max": 0.0,
        "L_offset": 0.00,
        "R_offset": 0.00
    }
}


# =====================================
# 시작 설정
# =====================================
current_stage = 0
drive_mode = "START_LEFT"
stage_start_time = time.time()
roundabout_direction = None

# 시작 방향 표지판 카운트
left_sign_count = 0
right_sign_count = 0

# 이벤트 상태 / 쿨타임 / 횟수
event_mode = None
event_start_time = 0.0
event_cooldown = 15.0
last_event_time = -999.0
sinho_count = 0
stop_count = 0
slow_count = 0

# RIGHT 경로에서만 사용하는 Rover 추적 상태
rover_seen = False
rover_max_area_ratio = 0.0
rover_away_count = 0
rover_missing_count = 0

print(
    f"[START] Stage: {current_stage} "
    f"| Mode: {drive_mode} "
    f"| Direction: {roundabout_direction}"
)


# =====================================
# 비동기 제어 / 차량 제어
# =====================================
def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({
            'T': 1,
            'L': L,
            'R': R
        })

    threading.Thread(target=worker, daemon=True).start()


def clip(val, max_val):
    return max(min(val, max_val), -max_val)


def update_vehicle_motion(steering, speed, turn_strength, L_offset, R_offset):
    steering = clip(steering, MAX_STEER)

    L = speed - steering * turn_strength
    R = speed + steering * turn_strength

    L += L_offset
    R += R_offset

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    # 기존 코드의 실제 모터 전송 순서를 그대로 유지
    send_control_async(R, L)

    print(
        f"[AUTO] Mode: {drive_mode:<35} "
        f"Steer: {steering:+.2f} "
        f"L_calc: {L:+.2f} R_calc: {R:+.2f} "
        f"Sent_L: {R:+.2f} Sent_R: {L:+.2f}"
    )


# =====================================
# 표지판 / 이벤트 탐지
# =====================================
def detect_sign(frame):
    results = yolo_model.predict(
        source=frame,
        verbose=False
    )

    detected = None
    side_detected = False

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = yolo_model.names[cls_id]
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            box_width = x2 - x1
            box_height = y2 - y1

            if (
                label == "Left"
                and box_width >= SIGN_CONFIG["LEFT"]["min_width"]
                and conf >= SIGN_CONFIG["LEFT"]["conf"]
            ):
                detected = "LEFT"

            elif (
                label == "Right"
                and box_width >= SIGN_CONFIG["RIGHT"]["min_width"]
                and conf >= SIGN_CONFIG["RIGHT"]["conf"]
            ):
                detected = "RIGHT"

            elif (
                label == "Sinho"
                and box_height >= SIGN_CONFIG["SINHO"]["min_height"]
                and conf >= SIGN_CONFIG["SINHO"]["conf"]
            ):
                detected = "SINHO"

            elif (
                label == "Slow"
                and box_width >= SIGN_CONFIG["SLOW"]["min_width"]
                and conf >= SIGN_CONFIG["SLOW"]["conf"]
            ):
                detected = "SLOW"

            elif (
                label == "Stop"
                and box_width >= SIGN_CONFIG["STOP"]["min_width"]
                and conf >= SIGN_CONFIG["STOP"]["conf"]
            ):
                detected = "STOP"


    return detected, side_detected


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
    global rover_missing_count

    rover_seen = False
    rover_max_area_ratio = 0.0
    rover_away_count = 0
    rover_missing_count = 0


def detect_rover_area(frame):
    """
    rover.pt가 Rover 단일 클래스 검출 모델이라고 가정한다.
    여러 box가 잡히면 가장 큰 bbox를 접근 차량으로 사용한다.
    """
    results = rover_model.predict(
        source=frame,
        conf=ROVER_CONF,
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
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()

            area_ratio = float(
                ((x2 - x1) * (y2 - y1)) / frame_area
            )

            if area_ratio > largest_area_ratio:
                largest_area_ratio = area_ratio
                largest_conf = conf

    return largest_area_ratio, largest_conf


def check_rover_clear(frame):
    """
    Rover 출발 대기 모드에서만 실행된다.

    confidence 0.70 이상인 Rover bbox만 비교한다.
    bbox가 커지는 동안에는 접근 중으로 보고 정지한다.
    관측된 최대 bbox보다 충분히 작아지는 상태가 연속 확인될 때만 출발한다.

    confidence 0.70 이상인 Rover가 현재 프레임에 없으면
    그 프레임은 비교에 사용하지 않고 계속 정지한다.
    """
    global rover_seen
    global rover_max_area_ratio
    global rover_away_count
    global rover_missing_count

    area_ratio, conf = detect_rover_area(frame)

    # confidence 0.70 이상인 유효 Rover 검출이 없으면 계속 대기
    if area_ratio < ROVER_MIN_AREA_RATIO:
        print(
            "[ROVER] WAIT - "
            "No Rover detection above confidence 0.70"
        )
        return False

    rover_seen = True
    rover_missing_count = 0

    # bbox가 이전보다 커지면 아직 접근 중 또는 가장 가까워지는 중
    if area_ratio > rover_max_area_ratio:
        rover_max_area_ratio = area_ratio
        rover_away_count = 0

        print(
            f"[ROVER] APPROACHING/CLOSE "
            f"conf={conf:.2f} area={area_ratio:.4f} "
            f"max={rover_max_area_ratio:.4f}"
        )
        return False

    # 가장 크게 보였던 bbox보다 충분히 작아졌으면 멀어지는 중
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
def update_stage(now, detected_sign, circle_detected, rover_clear):
    global current_stage
    global stage_start_time
    global drive_mode
    global roundabout_direction
    global left_sign_count
    global right_sign_count

    elapsed = now - stage_start_time

    # =================================
    # STAGE 0: 시작 주행 중 Left/Right 횟수 비교
    # =================================
    if current_stage == 0:

        if detected_sign == "LEFT":
            left_sign_count += 1
            print(f"[LEFT COUNT] {left_sign_count}")

        elif detected_sign == "RIGHT":
            right_sign_count += 1
            print(f"[RIGHT COUNT] {right_sign_count}")

        if elapsed > STAGE_PARAMS["START_LEFT"]["duration"]:

            print(
                f"[FINAL COUNT] "
                f"L={left_sign_count} R={right_sign_count}"
            )

            # =================================
            # RIGHT 우세: Rover가 멀어질 때까지 정지
            # =================================
            if right_sign_count > left_sign_count:
                roundabout_direction = "RIGHT"
                current_stage = 10
                drive_mode = "WAIT_ROVER_CLEAR"
                stage_start_time = time.time()

                reset_rover_tracking()

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

                print(
                    "[STAGE] RIGHT SELECTED "
                    "-> WAIT_ROVER_CLEAR "
                    "(rover.pt enabled)"
                )

            # =================================
            # LEFT 우세 또는 동점: 기존처럼 2초 정지 후 출발
            # =================================
            else:
                roundabout_direction = "LEFT"
                current_stage = 22
                drive_mode = "ROUNDABOUT_LEFT_ENTER"

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

                print(
                    "[STAGE] LEFT SELECTED "
                    "-> STOP 2.0s "
                    "-> ROUNDABOUT_LEFT_ENTER "
                    "(rover.pt not used)"
                )

                time.sleep(2.0)
                stage_start_time = time.time()

    # =================================
    # STAGE 10: RIGHT 경로 전용 Rover 통과 대기
    # =================================
    elif current_stage == 10:

        if rover_clear:
            current_stage = 21
            stage_start_time = time.time()
            drive_mode = "ROUNDABOUT_RIGHT_ENTER"

            print(
                "[STAGE] ROVER CLEAR "
                "-> ROUNDABOUT_RIGHT_ENTER"
            )

    # =================================
    # STAGE 21: 첫번째 RIGHT 루트 진입 종료 후 일반 주행
    # =================================
    elif current_stage == 21:

        elapsed = time.time() - stage_start_time

        if elapsed > STAGE_PARAMS["ROUNDABOUT_RIGHT_ENTER"]["duration"]:
            current_stage = 3
            stage_start_time = time.time()
            drive_mode = "NORMAL_LEFT"

    # =================================
    # STAGE 22: 첫번째 LEFT 루트
    # =================================
    elif current_stage == 22:

        elapsed = time.time() - stage_start_time

        if drive_mode == "ROUNDABOUT_LEFT_ENTER":

            if elapsed > STAGE_PARAMS["ROUNDABOUT_LEFT_ENTER"]["duration"]:
                stage_start_time = time.time()
                drive_mode = "ROTATION_ONLY"

        elif drive_mode == "ROTATION_ONLY":

            if elapsed > STAGE_PARAMS["ROTATION_ONLY"]["duration"]:
                current_stage = 3
                stage_start_time = time.time()
                drive_mode = "NORMAL_RIGHT"

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

    # =================================
    # STAGE 3: 일반 주행 중 두번째 회전교차로 기준 객체(Circle) 대기
    # =================================
    elif current_stage == 3:

        if circle_detected:

            # =================================
            # 처음 표지판에서 LEFT가 선택된 경우:
            # Circle이 충분히 크게 탐지되면 우선 정지 후 Rover 통과 대기
            # =================================
            if roundabout_direction == "LEFT":
                current_stage = 40
                stage_start_time = time.time()
                drive_mode = "WAIT_SECOND_LEFT_ROVER_CLEAR"

                reset_rover_tracking()

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

                print(
                    "[STAGE] CIRCLE DETECTED AFTER FIRST LEFT "
                    "-> WAIT_SECOND_LEFT_ROVER_CLEAR "
                    "(STOP + rover.pt enabled)"
                )

            # 기존 RIGHT 분기는 그대로 유지
            elif roundabout_direction == "RIGHT":
                current_stage = 42
                stage_start_time = time.time()
                drive_mode = "SECOND_ROUNDABOUT_RIGHT_ENTER"

    # =================================
    # STAGE 40: 처음 LEFT 경로였을 때 Rover가 멀어질 때까지 정지
    # =================================
    elif current_stage == 40:

        if rover_clear:
            current_stage = 41
            stage_start_time = time.time()
            drive_mode = "SECOND_ROUNDABOUT_LEFT"

            print(
                "[STAGE] ROVER CLEAR AFTER FIRST LEFT "
                "-> SECOND_ROUNDABOUT_LEFT "
                "(FIXED STEERING 4.0s)"
            )

    # =================================
    # STAGE 41: 강제 조향 4초 후 직진 10초, 이후 정지
    # =================================
    elif current_stage == 41:

        elapsed = time.time() - stage_start_time

        if drive_mode == "SECOND_ROUNDABOUT_LEFT":

            if elapsed > STAGE_PARAMS["SECOND_ROUNDABOUT_LEFT"]["duration"]:
                stage_start_time = time.time()
                drive_mode = "SECOND_LEFT_STRAIGHT"

                print(
                    "[MODE] SECOND_ROUNDABOUT_LEFT FINISHED "
                    "-> SECOND_LEFT_STRAIGHT (10.0s)"
                )

        elif drive_mode == "SECOND_LEFT_STRAIGHT":

            if elapsed > STAGE_PARAMS["SECOND_LEFT_STRAIGHT"]["duration"]:
                current_stage = 99
                stage_start_time = time.time()
                drive_mode = "MISSION_STOP"

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

                print("[MISSION END] SECOND LEFT ROUTE FINISHED -> STOP")

    elif current_stage == 42:

        elapsed = time.time() - stage_start_time

        # =================================
        # 1. Circle 감지 후 강제 조향 4초
        # =================================
        if drive_mode == "SECOND_ROUNDABOUT_RIGHT_ENTER":

            if elapsed > STAGE_PARAMS["SECOND_ROUNDABOUT_RIGHT_ENTER"]["duration"]:
                stage_start_time = time.time()
                drive_mode = "SECOND_ROTATION_ONLY"

                print(
                    "[MODE] SECOND_ROUNDABOUT_RIGHT_ENTER FINISHED "
                    "-> SECOND_ROTATION_ONLY (10.0s)"
                )

        # =================================
        # 2. rotation_only_model 10초
        # =================================
        elif drive_mode == "SECOND_ROTATION_ONLY":

            if elapsed > STAGE_PARAMS["SECOND_ROTATION_ONLY"]["duration"]:
                stage_start_time = time.time()
                drive_mode = "SECOND_RIGHT_FIXED_TURN"

                print(
                    "[MODE] SECOND_ROTATION_ONLY FINISHED "
                    "-> SECOND_RIGHT_FIXED_TURN: FIXED RIGHT STEERING (3.0s)"
                )

        # =================================
        # 3. 오른쪽 고정 조향 명령 3초
        # =================================
        elif drive_mode == "SECOND_RIGHT_FIXED_TURN":

            if elapsed > STAGE_PARAMS["SECOND_RIGHT_FIXED_TURN"]["duration"]:
                stage_start_time = time.time()
                drive_mode = "SECOND_RIGHT_STRAIGHT"

                print(
                    "[MODE] SECOND_RIGHT_FIXED_TURN FINISHED "
                    "-> SECOND_RIGHT_STRAIGHT (10.0s)"
                )

        # =================================
        # 4. 모델 없이 직진 명령 10초 후 정지
        # =================================
        elif drive_mode == "SECOND_RIGHT_STRAIGHT":

            if elapsed > STAGE_PARAMS["SECOND_RIGHT_STRAIGHT"]["duration"]:
                current_stage = 99
                stage_start_time = time.time()
                drive_mode = "MISSION_STOP"

                base.base_json_ctrl({
                    'T': 1,
                    'L': 0.0,
                    'R': 0.0
                })

                print("[MISSION END] SECOND RIGHT ROUTE FINISHED -> STOP")


# =====================================
# MAIN LOOP
# =====================================
try:
    while True:
        frame = camera.read()

        height, width = frame.shape[:2]
        center_x = width / 2
        now = time.time()

        detected_sign, _ = detect_sign(frame)

        # Circle은 두 번째 회전교차로 접근 판단이 필요한 일반 주행 구간에서만 탐지
        # 따라서 best_circle.pt가 불필요하게 매 프레임 실행되는 것을 줄인다.
        circle_detected = False

        if current_stage == 3:
            circle_detected = detect_circle(frame)

        # Rover는 필요한 정지 대기 구간에서만 탐지
        rover_clear = False

        if drive_mode in (
            "WAIT_ROVER_CLEAR",
            "WAIT_SECOND_LEFT_ROVER_CLEAR"
        ):
            rover_clear = check_rover_clear(frame)

        update_stage(
            now,
            detected_sign,
            circle_detected,
            rover_clear
        )

        # =================================
        # 일반 주행 중 이벤트 감지
        # =================================
        if drive_mode in ("NORMAL_LEFT", "NORMAL_RIGHT"):

            if (
                event_mode is None
                and now - last_event_time > event_cooldown
            ):

                if detected_sign == "SINHO" and sinho_count < 2:
                    event_mode = "SINHO"
                    event_start_time = now
                    last_event_time = now
                    sinho_count += 1
                    print("[EVENT] SINHO")

                elif detected_sign == "STOP" and stop_count < 2:
                    event_mode = "STOP"
                    event_start_time = now
                    last_event_time = now
                    stop_count += 1
                    print("[EVENT] STOP")

                elif detected_sign == "SLOW" and slow_count < 2:
                    event_mode = "SLOW"
                    event_start_time = now
                    last_event_time = now
                    slow_count += 1
                    print("[EVENT] SLOW")

        params = STAGE_PARAMS[drive_mode]
        current_speed = params["speed"]

        # =================================
        # 현재 모드에 따른 steering 계산
        # =================================
        if params["use_fixed_steering"]:
            steering = params["fixed_steering"]

        else:
            input_tensor = preprocess(frame)

            with torch.no_grad():
                output = params["model"](input_tensor)

            x, y = output[0].detach().cpu().numpy()

            pred_x = (x / 2 + 0.5) * width
            error = pred_x - center_x
            error_norm = error / center_x

            steering = params["Kp"] * error_norm + params["bias"]

            # 마지막 코너 boost: NORMAL_RIGHT 전용
            if drive_mode == "NORMAL_RIGHT":
                elapsed_stage_time = now - stage_start_time

                if elapsed_stage_time > 30.0 and 0.09 <= steering <= 0.15:
                    steering += 0.035

            # deadzone
            if params["use_deadzone"]:
                if params["deadzone_min"] <= steering <= params["deadzone_max"]:
                    steering = 0.0

        steering = clip(steering, MAX_STEER)

        # =================================
        # EVENT OVERRIDE
        # =================================
        if event_mode == "SINHO":

            if now - event_start_time < 2.0:
                steering = 0.0
                current_speed = 0.0
            else:
                event_mode = None

        elif event_mode == "STOP":

            if now - event_start_time < 3.0:
                steering = 0.0
                current_speed = 0.0
            else:
                event_mode = None

        elif event_mode == "SLOW":

            if now - event_start_time < 5.0:
                current_speed *= 0.5
            else:
                event_mode = None

        # =================================
        # 차량 제어
        # =================================
        update_vehicle_motion(
            steering,
            current_speed,
            params["turn_strength"],
            params["L_offset"],
            params["R_offset"]
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
