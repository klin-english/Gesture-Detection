import warnings
import cv2, mediapipe as mp, os, time as t, subprocess, shutil, glob
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import platform

warnings.filterwarnings("ignore",
                        message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
                        category=UserWarning,
                        module=r"google\.protobuf")

# ---- Platform flags ----
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_ARM = "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower()
ON_PI = IS_LINUX and IS_ARM

# ---- Windows-only imports ----
if IS_WINDOWS:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    import screen_brightness_control as sbc
else:
    print("[Info] Non-Windows OS detected — using Raspberry Pi/Unix fallbacks for audio/brightness where possible.")

# ---- Camera selection + watchdog (USB Creative webcam / UVC preferred, Pi fallback) ----
read_frame = None
cap = None
picam2 = None
video_index = None
MAX_FAILS = 10
fail_count = 0
VIDEO_INDICES = list(range(0, 5))  # scan /dev/video0..4
CAM_W = 640
CAM_H = 480
CAM_FPS = 30
LOOP_DELAY = 0.5  # safety delay in main loop (seconds)

def try_open_index(idx):
    """Try to open a given video index and warm it up. Return cv2.VideoCapture or None."""
    try:
        api = cv2.CAP_V4L2 if IS_LINUX else 0
        c = cv2.VideoCapture(idx, api)
        if not c.isOpened():
            try:
                c.release()
            except Exception:
                pass
            return None
        # Force stable webcam settings (MJPG often helps UVC stability)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        c.set(cv2.CAP_PROP_FPS, CAM_FPS)
        # Warm up a couple frames to ensure it's delivering
        ok, frame = False, None
        for _ in range(3):
            ok, frame = c.read()
            if ok and frame is not None:
                break
            t.sleep(0.05)
        if not ok or frame is None:
            try:
                c.release()
            except Exception:
                pass
            return None
        return c
    except Exception:
        return None

def init_camera():
    """Initialize a working camera: prefer USB UVC (Creative) webcams, fall back to Picamera2 on Pi."""
    global read_frame, cap, picam2, fail_count, video_index
    fail_count = 0
    video_index = None

    # Try to find a working USB webcam first
    for idx in VIDEO_INDICES:
        c = try_open_index(idx)
        if c:
            cap = c
            video_index = idx
            print(f"[Camera] USB webcam active at /dev/video{idx} (MJPG, {CAM_W}x{CAM_H}@{CAM_FPS}fps).")
            break

    # If none found and we're on a Pi, try Picamera2 (Global Shutter)
    if cap is None and ON_PI:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": (CAM_W, CAM_H)}
            ))
            picam2.start()

            def _read_picam2():
                global fail_count
                try:
                    frame = picam2.capture_array()
                    if frame is None:
                        fail_count += 1
                        return False, None
                    # Convert to BGR for OpenCV
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    fail_count = 0
                    return True, frame_bgr
                except Exception as e:
                    print(f"[Camera Error] Picamera2 read failed: {e}")
                    fail_count += 1
                    return False, None

            read_frame = _read_picam2
            print(f"[Camera] Picamera2 (Pi camera) active ({CAM_W}x{CAM_H}).")
            return
        except Exception as e:
            print(f"[Warn] Picamera2 unavailable: {e}. Continuing to try USB webcams...")

    if cap is None:
        raise RuntimeError("[Error] Cannot access any camera device (no USB webcam and no Pi camera).")

    # Define USB read function with auto-reopen logic
    def _read_usb():
        global cap, fail_count, video_index
        # If camera handle invalid, try to reopen the same index, otherwise rescan
        if cap is None or not getattr(cap, "isOpened", lambda: False)():
            print("[Camera] Webcam handle is closed; attempting to reopen...")
            # First try previous index
            if video_index is not None:
                c = try_open_index(video_index)
                if c:
                    cap = c
                else:
                    # scan for any other working index
                    found = False
                    for idx in VIDEO_INDICES:
                        c2 = try_open_index(idx)
                        if c2:
                            cap = c2
                            video_index = idx
                            found = True
                            print(f"[Camera] Re-opened at /dev/video{idx}.")
                            break
                    if not found:
                        print("[Camera] No webcam found during re-scan.")
                        return False, None

        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            # Try a quick re-open on repeated failures
            if fail_count >= MAX_FAILS:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                # rescanning indices
                found = False
                for idx in VIDEO_INDICES:
                    c2 = try_open_index(idx)
                    if c2:
                        cap = c2
                        video_index = idx
                        found = True
                        print(f"[Camera] Reinitialized to /dev/video{idx}.")
                        break
                fail_count = 0
                if not found:
                    print("[Camera] Rescan found no working USB webcam.")
                    return False, None
            return False, None
        fail_count = 0
        return True, frame

    read_frame = _read_usb

# initialize camera
init_camera()

# ---- MediaPipe ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---- Load ASL model if present ----
model_path = Path("model/asl_knn_model.joblib")
encoder_path = Path("model/label_encoder.joblib")
if model_path.exists() and encoder_path.exists():
    try:
        asl_model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        print("[Model] ASL recognition model loaded.")
    except Exception as e:
        asl_model = None
        label_encoder = None
        print(f"[Warning] Failed to load ASL model files: {e}")
else:
    asl_model = None
    label_encoder = None
    print("[Warning] No ASL model found. ASL recognition will be disabled.")

# ---- Pi/Unix helpers ----
def _run_quiet(args):
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

def _pi_has_backlight():
    return bool(glob.glob("/sys/class/backlight/*/brightness"))

# ---- Actions (Windows + Pi fallbacks) ----
def brightness_up(step=10):
    if IS_WINDOWS:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(min(100, current + step), display=0)
            print(f"[Action] Brightness increased to {min(100, current + step)}%")
        except Exception as e:
            print(f"[Error] Brightness control failed: {e}")
    elif ON_PI:
        if shutil.which("brightnessctl"):
            _run_quiet(["brightnessctl", "set", f"+{step}%"])
            print(f"[Action] brightnessctl +{step}%")
        elif _pi_has_backlight():
            try:
                path = glob.glob("/sys/class/backlight/*")[0]
                with open(os.path.join(path, "max_brightness")) as f: mx = int(f.read().strip())
                with open(os.path.join(path, "brightness")) as f: cur = int(f.read().strip())
                new = min(mx, cur + max(1, mx * step // 100))
                with open(os.path.join(path, "brightness"), "w") as f: f.write(str(new))
                print("[Action] Backlight increased.")
            except PermissionError:
                print("[Info] Backlight requires sudo. Skipping.")
        else:
            print("[Info] No controllable backlight detected.")
    else:
        print("[Info] Brightness control not supported on this OS.")

def brightness_down(step=10):
    if IS_WINDOWS:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(max(0, current - step), display=0)
            print(f"[Action] Brightness decreased to {max(0, current - step)}%")
        except Exception as e:
            print(f"[Error] Brightness control failed: {e}")
    elif ON_PI:
        if shutil.which("brightnessctl"):
            _run_quiet(["brightnessctl", "set", f"-{step}%"])
            print(f"[Action] brightnessctl -{step}%")
        elif _pi_has_backlight():
            try:
                path = glob.glob("/sys/class/backlight/*")[0]
                with open(os.path.join(path, "max_brightness")) as f: mx = int(f.read().strip())
                with open(os.path.join(path, "brightness")) as f: cur = int(f.read().strip())
                new = max(0, cur - max(1, mx * step // 100))
                with open(os.path.join(path, "brightness"), "w") as f: f.write(str(new))
                print("[Action] Backlight decreased.")
            except PermissionError:
                print("[Info] Backlight requires sudo. Skipping.")
        else:
            print("[Info] No controllable backlight detected.")
    else:
        print("[Info] Brightness control not supported on this OS.")

def toggle_mute():
    if IS_WINDOWS:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(0 if volume.GetMute() else 1, None)
    elif ON_PI:
        _run_quiet(["amixer", "-q", "sset", "Master", "toggle"])
        print("[Action] ALSA mute toggled.")
    else:
        print("[Info] Mute toggle not supported on this OS.")

def volume_up(step=10):
    if IS_WINDOWS:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        cur = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(min(1.0, cur + step/100.0), None)
    elif ON_PI:
        _run_quiet(["amixer", "-q", "sset", "Master", f"{step}%+"])
        print(f"[Action] ALSA volume +{step}%.")
    else:
        print("[Info] Volume up not supported on this OS.")

def volume_down(step=10):
    if IS_WINDOWS:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        cur = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(max(0.0, cur - step/100.0), None)
    elif ON_PI:
        _run_quiet(["amixer", "-q", "sset", "Master", f"{step}%-"])
        print(f"[Action] ALSA volume -{step}%.")
    else:
        print("[Info] Volume down not supported on this OS.")

# ---- Gesture Actions ----
def do_open_palm_action():
    print("[Action] Turning lights ON (simulated).")

def do_pointing_action():
    print("[Action] You pointed — increasing volume by 10.")
    volume_up()

def do_peace_action():
    print("[Action] Peace gesture — muting audio...")
    toggle_mute()

def do_bird_action():
    print("[Action] Middle finger — rude! Logging it as a joke.")

def do_phone_action():
    print("[Action] Phone - lowering the brightness by 10.")
    brightness_down()

def do_thumbs_up_action():
    print("[Action] Thumbs up — upping the brightness by 10.")
    brightness_up()

def do_other_bird_action():
    print("[Action] Pinky - Lowering volume by 10.")
    volume_down()

# ---- State ----
active_read = False
cooldown_secs = 3.0
last_event_time = 0
last_printed = None
command_cooldown_secs = 1.5
last_command_time = 0
asl_active = False
last_asl_letter = None
asl_start_time = 0

# ---- Main loop ----
with mp_hands.Hands(max_num_hands=1,
                    model_complexity=0,     # lighter model helps on Pi
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    print("Press 'q' to quit.")

    while True:
        ok, frame = read_frame()
        if not ok or frame is None:
            # camera lost frame; wait a little and continue (watchdog in read_frame will try to recover)
            t.sleep(LOOP_DELAY)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "Unknown"
        now = t.time()
        if now - last_command_time < command_cooldown_secs:
            cv2.imshow('Gesture Recognition', frame)
            # small delay to reduce CPU usage
            t.sleep(LOOP_DELAY)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if results.multi_hand_landmarks:
            for lm_set in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)
                lm = lm_set.landmark

                # ASL (optional)
                if asl_model and label_encoder and asl_active and (now - asl_start_time > 1.0):
                    try:
                        asl_features = [v for pt in lm_set.landmark for v in (pt.x, pt.y, pt.z)]
                        prediction = asl_model.predict([asl_features])[0]
                        predicted_sign = label_encoder.inverse_transform([prediction])[0]
                        if predicted_sign != last_asl_letter:
                            print(f"[ASL] Detected: {predicted_sign}")
                            last_asl_letter = predicted_sign
                        cv2.putText(frame, f"ASL: {predicted_sign}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                    except Exception as e:
                        print(f"[ASL Error] {e}")

                tips = [8, 12, 16, 20]
                # fingers: True = extended/up, False = down
                fingers = [(lm[p].y < lm[p - 2].y) for p in tips]
                # thumb direction check:
                try:
                    thumb_up = lm[4].x > lm[3].x if lm[17].x < lm[0].x else lm[4].x < lm[3].x
                except Exception:
                    thumb_up = False
                fingers.insert(0, thumb_up)

                if fingers == [False]*5 and not active_read:
                    gesture = "Fist"
                    active_read = True
                    last_event_time = now
                    print("Fist detected – command mode will start in "
                          f"{cooldown_secs:.0f} s…")
                    break

                if active_read and now - last_event_time < cooldown_secs:
                    break

                if active_read and not asl_active:
                    if fingers == [True]*5:
                        gesture, active_read = "Open Palm", False
                        do_open_palm_action(); last_command_time = now
                    elif fingers == [False, True, False, False, False]:
                        gesture, active_read = "Pointing", False
                        do_pointing_action(); last_command_time = now
                    elif fingers == [False, True, True, False, False]:
                        gesture, active_read = "Peace", False
                        do_peace_action(); last_command_time = now
                    elif fingers == [False, False, True, False, False]:
                        gesture, active_read = "Bird", False
                        do_bird_action(); last_command_time = now
                    elif fingers == [True, False, False, False, True]:
                        gesture, active_read = "Phone", False
                        do_phone_action(); last_command_time = now
                    elif fingers == [True, False, False, False, False]:
                        gesture, active_read = "Thumbs Up", False
                        do_thumbs_up_action(); last_command_time = now
                    elif fingers == [False, False, False, False, True]:
                        gesture, active_read = "Other Bird", False
                        do_other_bird_action(); last_command_time = now
                    elif fingers == [True, True, True, False, False]:
                        # avoid unpacking errors and guard ASL model presence
                        gesture = "OK"
                        active_read = False
                        if asl_model is not None and label_encoder is not None:
                            asl_active = True
                            asl_start_time = now
                            last_command_time = now
                            print("[Mode] ASL detection activated.")
                        else:
                            print("[Warning] ASL model not loaded — skipping ASL mode.")
        else:
            if asl_active:
                print("[Mode] ASL detection stopped — no hand detected.")
            asl_active = False

        if gesture != "Unknown" and gesture != last_printed:
            print(f"Gesture: {gesture}")
            last_printed = gesture

        cv2.imshow('Gesture Recognition', frame)
        # small sleep to reduce CPU and USB load
        t.sleep(LOOP_DELAY)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---- Cleanup ----
try:
    if cap is not None:
        cap.release()
except Exception:
    pass
try:
    if picam2 is not None:
        picam2.stop()
except Exception:
    pass
cv2.destroyAllWindows()
