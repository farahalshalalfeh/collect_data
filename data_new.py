import cv2
import os
import numpy as np
import mediapipe as mp

# =====================================
# Mediapipe Setup
# =====================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

def extract_keypoints(results):
    if not results.multi_hand_landmarks:
        return np.zeros(42)
    hand = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand.landmark:
        keypoints.extend([lm.x, lm.y])
    return np.array(keypoints)

# =====================================
# Image Enhancement
# =====================================
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced

# =====================================
# Classes (28 → 34)
# =====================================
BASE_CLASS_ID = 28

arabic_letters = ['ئ', 'ؤ', 'ء', 'ة', 'ى', 'لا', 'أ']

display_names = {
    'ئ': 'HAMZA ON YA',
    'ؤ': 'HAMZA ON WAW',
    'ء': 'HAMZA',
    'ة': 'TA MARBUTA',
    'ى': 'ALIF MAQSURA',
    'لا': 'LAM ALIF',
    'أ': 'ALIF WITH HAMZA ABOVE'
}

actions = np.arange(BASE_CLASS_ID, BASE_CLASS_ID + len(arabic_letters))  # 28 → 34

# =====================================
# Paths & Settings
# =====================================
DATA_PATH = os.path.join("MP_Data", "Alphabets")   # keypoints فقط
VIDEO_PATH = "Videos"                             # الفيديوهات برّه

TARGET_FPS = 30
SECONDS_PER_SEQUENCE = 3
sequence_length = TARGET_FPS * SECONDS_PER_SEQUENCE
videos_per_person = 25

# =====================================
# Helper
# =====================================
def get_next_sequence_index(class_id):
    folder = os.path.join(DATA_PATH, str(class_id))
    os.makedirs(folder, exist_ok=True)
    nums = [int(f) for f in os.listdir(folder) if f.isdigit()]
    return max(nums) + 1 if nums else 0

# =====================================
# Webcam
# =====================================
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for action in actions:

        idx = action - BASE_CLASS_ID
        letter = arabic_letters[idx]
        eng_name = display_names[letter]

        print(f"\nClass {action} → Letter: {letter} ({eng_name})")

        start_seq = get_next_sequence_index(action)

        for sequence in range(start_seq, videos_per_person):

            print(f"Recording video {sequence}")
            print("ENTER = start | SPACE = pause | Q = quit")

            video_folder = os.path.join(VIDEO_PATH, str(action))
            os.makedirs(video_folder, exist_ok=True)

            video_path = os.path.join(video_folder, f"{sequence}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                TARGET_FPS,
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            # -------- WAIT FOR ENTER --------
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                enhanced = enhance_image(frame)
                image, results = mediapipe_detection(enhanced, hands)
                draw_styled_landmarks(image, results)

                cv2.putText(image, f'{eng_name} | Video {sequence}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(image, 'Press ENTER to start | Q to quit',
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imshow("OpenCV Feed", image)

                key = cv2.waitKey(1) & 0xFF
                if key == 13:
                    break
                if key == ord('q'):
                    video_writer.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit()

            # -------- RECORDING --------
            frame_num = 0

            while frame_num < sequence_length:

                ret, frame = cap.read()
                if not ret:
                    continue

                enhanced = enhance_image(frame)
                image, results = mediapipe_detection(enhanced, hands)
                draw_styled_landmarks(image, results)

                cv2.imshow("OpenCV Feed", image)

                if cv2.waitKey(int(1000 / TARGET_FPS)) & 0xFF == ord('q'):
                    video_writer.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit()

                video_writer.write(image)

                kp = extract_keypoints(results)
                kp_path = os.path.join(DATA_PATH, str(action), str(sequence))
                os.makedirs(kp_path, exist_ok=True)
                np.save(os.path.join(kp_path, str(frame_num)), kp)

                frame_num += 1

            video_writer.release()

cap.release()
cv2.destroyAllWindows()
