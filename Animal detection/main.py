
import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os
import cloudinary
import cloudinary.uploader
import json

# ---------------- CLOUDINARY CONFIG ----------------
cloudinary.config(
    cloud_name="dtnro3tvp",
    api_key="959995371738249",
    api_secret="GTff-AFTNs2JAfBf9npTTYth35w"
)

CLOUD_FOLDER = "animal_detections"
#------------ CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

ANIMAL_CLASSES = [
    'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe'
]
SAVE_FOLDER = "detections"
SAVE_INTERVAL = 3
LOG_FILE = "images.json"

# ----------------------------------------

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Initialize JSON file if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)


def save_image_url(url):
    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    data.append({
        "url": url,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_model():
    print("[INFO] Loading YOLOv8 model...")
    return YOLO(MODEL_PATH)


def run_webcam():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not working")
        return

    prev_time = time.time()
    last_saved_time = 0

    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in ANIMAL_CLASSES and conf > CONFIDENCE_THRESHOLD:
                    detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

        # 🔥 SAVE + UPLOAD
        current_time = time.time()
        if detected and (current_time - last_saved_time > SAVE_INTERVAL):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(SAVE_FOLDER, f"animal_{timestamp}.jpg")

            cv2.imwrite(filepath, frame)
            print(f"[INFO] Saved: {filepath}")

            try:
                upload = cloudinary.uploader.upload(filepath, folder=CLOUD_FOLDER)
                image_url = upload['secure_url']

                print(f"[☁️ UPLOADED] {image_url}")

                save_image_url(image_url)

            except Exception as e:
                print("[ERROR Upload]", e)

            last_saved_time = current_time

        # FPS
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()