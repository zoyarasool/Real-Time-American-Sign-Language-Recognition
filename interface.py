import cv2
import os
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

#config
IMG_SIZE = 64
DIGIT_THRESHOLD = 0.90
SMOOTHING_FRAMES = 12

#paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LABELS_DIR = os.path.join(ROOT_DIR, "lables")

ALPHABET_MODEL_PATH = os.path.join(MODELS_DIR, "asl_alphabets_model.h5")
DIGIT_MODEL_PATH    = os.path.join(MODELS_DIR, "asl_digits_model.h5")

#load models & labels
alphabet_model = load_model(ALPHABET_MODEL_PATH)
digit_model = load_model(DIGIT_MODEL_PATH)

with open(os.path.join(LABELS_DIR, "alphabet_labels.json")) as f:
    alphabet_map = json.load(f)

with open(os.path.join(LABELS_DIR, "digits_labels.json")) as f:
    digit_map = json.load(f)

alphabet_labels = {v: k.upper() for k, v in alphabet_map.items()}
digit_labels = {v: k for k, v in digit_map.items()}

print("Alphabet & Digit models loaded")

#mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#camera
cap = cv2.VideoCapture(0)

pred_queue = deque(maxlen=SMOOTHING_FRAMES)

#helpers
def preprocess_hand(hand_img):
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    h, w, _ = hand_img.shape
    size = max(h, w)

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = hand_img

    img = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "SHOW HAND"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in hand.landmark]
        ys = [int(lm.y * h) for lm in hand.landmark]

        x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
        y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)

        hand_img = frame[y1:y2, x1:x2]

        if hand_img.size != 0:
            inp = preprocess_hand(hand_img)

            a_pred = alphabet_model.predict(inp, verbose=0)[0]
            d_pred = digit_model.predict(inp, verbose=0)[0]

            pred_queue.append((a_pred, d_pred))

            avg_a = np.mean([p[0] for p in pred_queue], axis=0)
            avg_d = np.mean([p[1] for p in pred_queue], axis=0)

            a_idx, d_idx = np.argmax(avg_a), np.argmax(avg_d)
            a_conf, d_conf = np.max(avg_a), np.max(avg_d)

            if d_conf >= DIGIT_THRESHOLD:
                text = f"DIGIT {digit_labels[d_idx]} ({d_conf:.2f})"
            else:
                text = f"ALPHABET {alphabet_labels[a_idx]} ({a_conf:.2f})"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(frame, text, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("ASL Alphabet & Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
