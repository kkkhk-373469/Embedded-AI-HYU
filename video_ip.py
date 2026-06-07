from jetcam.csi_camera import CSICamera
from flask import Flask, Response
from ultralytics import YOLO

import cv2
import numpy as np

app = Flask(__name__)

# =========================
# YOLO 모델
# =========================

tire_model = YOLO("tire.pt")
sign_model = YOLO("sign.pt")

# =========================
# 카메라
# =========================

cam0 = CSICamera(
    capture_device=0,
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)

cam1 = CSICamera(
    capture_device=1,
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)

def generate():

    while True:

        frame0 = cam0.read()
        frame1 = cam1.read()

        if frame0 is None or frame1 is None:
            continue

        # =====================
        # CAM0 추론
        # =====================

        tire_results0 = tire_model(
            frame0,
            imgsz=640,
            conf=0.6,
            verbose=False
        )

        sign_results0 = sign_model(
            frame0,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        annotated0 = frame0.copy()

        # Tire
        if tire_results0[0].boxes is not None:

            for box in tire_results0[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = (
                    f"{tire_model.names[cls]}"
                    f" {conf:.2f}"
                )

                cv2.rectangle(
                    annotated0,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    annotated0,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        # Sign
        if sign_results0[0].boxes is not None:

            for box in sign_results0[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = (
                    f"{sign_model.names[cls]}"
                    f" {conf:.2f}"
                )

                cv2.rectangle(
                    annotated0,
                    (x1, y1),
                    (x2, y2),
                    (255,0,0),
                    2
                )

                cv2.putText(
                    annotated0,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,0),
                    2
                )

        cv2.putText(
            annotated0,
            "CAM0",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # =====================
        # CAM1 추론
        # =====================

        tire_results1 = tire_model(
            frame1,
            imgsz=640,
            conf=0.6,
            verbose=False
        )

        sign_results1 = sign_model(
            frame1,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        annotated1 = frame1.copy()

        # Tire
        if tire_results1[0].boxes is not None:

            for box in tire_results1[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = (
                    f"{tire_model.names[cls]}"
                    f" {conf:.2f}"
                )

                cv2.rectangle(
                    annotated1,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    annotated1,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        # Sign
        if sign_results1[0].boxes is not None:

            for box in sign_results1[0].boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = (
                    f"{sign_model.names[cls]}"
                    f" {conf:.2f}"
                )

                cv2.rectangle(
                    annotated1,
                    (x1, y1),
                    (x2, y2),
                    (255,0,0),
                    2
                )

                cv2.putText(
                    annotated1,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,0),
                    2
                )

        cv2.putText(
            annotated1,
            "CAM1",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # =====================
        # 두 화면 합치기
        # =====================

        combined = np.hstack(
            (annotated0, annotated1)
        )

        ret, buffer = cv2.imencode(
            ".jpg",
            combined
        )

        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )

@app.route('/')
def home():

    return """
    <html>
        <body>
            <h2>Dual Camera Detection</h2>
            <img src="/video">
        </body>
    </html>
    """
@app.route('/video')
def video():

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# =========================
# 실행
# =========================

if __name__ == "__main__":

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
