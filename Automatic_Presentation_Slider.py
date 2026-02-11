import cv2
import mediapipe as mp
import pyautogui
import time

# ------------------ SETUP ------------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

last_action_time = 0
delay = 1.2   # seconds (prevents multiple triggers)

# ------------------ FIST DETECTION ------------------
def is_fist(hand):
    wrist = hand.landmark[0]
    tips = [8, 12, 16, 20]  # finger tips

    for tip in tips:
        if abs(hand.landmark[tip].y - wrist.y) > 0.08:
            return False
    return True

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # PALM (WRIST) POSITION
        palm = hand.landmark[0]
        x_px = int(palm.x * w)
        y_px = int(palm.y * h)

        # Draw palm point
        cv2.circle(frame, (x_px, y_px), 10, (0, 0, 255), -1)


        # ---------- LEFT / RIGHT SLIDE ----------
        if x_px < w // 3 and current_time - last_action_time > delay:
            pyautogui.press('left')
            print("PREVIOUS SLIDE")
            last_action_time = current_time

        elif x_px > 2 * w // 3 and current_time - last_action_time > delay:
            pyautogui.press('right')
            print("NEXT SLIDE")
            last_action_time = current_time

    # Draw region guides
    cv2.line(frame, (w//3, 0), (w//3, h), (0, 255, 0), 2)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), (0, 255, 0), 2)
    cv2.line(frame, (0, h//3), (w, h//3), (255, 0, 0), 2)

    cv2.imshow("Gesture Presentation Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()