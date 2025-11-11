import cv2
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("❌ MediaPipe not available for Python 3.13")
    print("💡 Solutions:")
    print("   1. Install Python 3.11/3.12 from python.org")
    print("   2. Use alternative hand tracking libraries")
    print("   3. Modify code to use OpenCV-based detection")
    MEDIAPIPE_AVAILABLE = False
import numpy as np
import json
import os
import pathlib
import platform

# ---- Platform / camera selection ----
IS_LINUX = platform.system() == "Linux"
IS_ARM = "arm" in platform.machine() or "aarch64" in platform.machine()
ON_PI = IS_LINUX and IS_ARM

read_frame = None
cap = None
picam2 = None

def init_camera():
    global read_frame, cap, picam2
    if ON_PI:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            # Use XRGB for cheap conversion to BGR for OpenCV drawing
            picam2.configure(picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": (640, 480)}
            ))
            picam2.start()

            def _read():
                frame = picam2.capture_array()
                # Convert BGRA->BGR if 4 channels
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame_bgr
            read_frame = _read
            print("[Camera] Picamera2 active (640x480).")
            return
        except ImportError:
            print("[Info] Picamera2 not available on this system. Using OpenCV fallback.")
        except Exception as e:
            print(f"[Warn] Picamera2 unavailable or failed: {e}. Falling back to OpenCV V4L2.")

    # Fallback: USB cam / desktop
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2 if IS_LINUX else 0)
    if not cap.isOpened():
        raise RuntimeError("[Error] Cannot access camera device.")
    read_frame = lambda: cap.read()
    print("[Camera] OpenCV VideoCapture active.")

init_camera()

# ---- MediaPipe Hands ----
if not MEDIAPIPE_AVAILABLE:
    print("❌ Cannot run ASL data collection without MediaPipe")
    print("💡 Please install Python 3.11/3.12 and recreate virtual environment")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,             # Lighter model for Raspberry Pi
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Output store
data = []

# Labels
sign_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Dataset path
json_path = "dataset/asl_data.json"
os.makedirs("dataset", exist_ok=True)

# Load existing
if pathlib.Path(json_path).exists():
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"[Loaded] Existing dataset with {len(data)} samples.")

print("[Instructions]")
print("Press a key (A–Z) to change label.")
print("Press SPACE to save a sample.")
print("Press ';' to quit and save.\n")

current_label = "A"

while True:
    ret, frame = read_frame()
    if not ret:
        print("[Error] Frame grab failed.")
        break

    frame = cv2.flip(frame, 1)  # mirror
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("ASL Data Collector (Pi/Desktop)", frame)
    key = cv2.waitKey(1)

    if key == -1:
        continue
    if key == ord(';'):
        break
    elif key == ord(' '):
        if landmark_list:
            sample = {"label": current_label, "landmarks": landmark_list}
            data.append(sample)
            print(f"[Saved] '{current_label}' — Total: {len(data)}")
        else:
            print("[Skip] No hand detected")
    elif 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()

if cap is not None:
    cap.release()
if picam2 is not None:
    picam2.stop()
cv2.destroyAllWindows()

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"[Complete] Saved {len(data)} total samples to '{json_path}'")
