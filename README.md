This project focused on enabling an autonomous Wave Rover to perceive its surroundings and execute specific driving missions using YOLO-based object detection and PID control.
<img width="1160" height="605" alt="스크린샷 2026-05-23 오후 5 06 20" src="https://github.com/user-attachments/assets/c38fb7a2-89ad-4da2-9795-15679561dbb2" />
<img width="1156" height="599" alt="스크린샷 2026-05-23 오후 5 06 25" src="https://github.com/user-attachments/assets/321ff9d1-2963-4d92-870b-e31cf63f3a1c" />
<img width="1158" height="601" alt="스크린샷 2026-05-23 오후 5 06 30" src="https://github.com/user-attachments/assets/c2604b63-9ab7-45d2-92b6-8e61f0dd9562" />
<img width="1158" height="590" alt="스크린샷 2026-05-23 오후 5 06 35" src="https://github.com/user-attachments/assets/9e31bd6f-ed23-46b0-9a37-335b804b10bf" />
# Automotive Embedded AI Autonomous Driving Project

카메라 기반 소형 자율주행 차량(Wave Rover)을 활용하여 차선 추종, 교통 표지판·신호등 인식, 회전교차로 경로 선택 및 차량 회피 주행을 구현하는 프로젝트입니다.  
CNN 기반 경로 중심점 예측 모델을 이용해 실시간 조향을 수행하고, YOLO 기반 객체 탐지 모델로 주행 상황을 판단하여 상황별 제어 모드로 전환합니다.

## Project Overview

본 프로젝트의 주행 코스는 회전교차로, 직선 교차로, 교통 표지판 및 NPC 차량이 포함된 복합 환경으로 구성되어 있습니다. 차량은 입력 영상으로부터 주행 경로를 추종하며, 표지판과 주변 객체를 인식하여 요구되는 미션을 수행합니다.

## Tasks

| Task | 내용 | 핵심 기능 |
|---|---|---|
| Task 1 | 회전교차로 주행 | 좌·우회전 표지판 인식 후 지정 경로 주행, 반시계 방향 회전교차로 통과 |
| Task 2 | 표지판 확인 후 서행 또는 정지 | STOP 표지판 감지 시 일시 정지 후 재출발, 보행자 표지판 감지 시 감속 주행 |
| Task 3 | 신호등 인식 및 직선 교차로 주행 | 신호등 상태 인식 후 정지 또는 교차로 통과 |
| Task 4 | 회전교차로 차량 회피 주행 | 회전교차로 내 NPC 차량 인식 및 충돌 회피 주행 |

## System Architecture

```text
Camera Input
    ↓
YOLO Object Detection
- Left / Right Sign
- STOP / Slow Sign
- Traffic Light
- Vehicle / Exit Object
    ↓
Driving Mode Decision
- Normal Lane Following
- Roundabout Entry / Rotation / Exit
- Stop / Slow Event
- Vehicle Avoidance
    ↓
CNN-Based Path Tracking Model
- Target Point Prediction (x, y)
    ↓
Steering Calculation
- Error between predicted point and image center
- Left / Right wheel velocity control
    ↓
Wave Rover Motion Control
