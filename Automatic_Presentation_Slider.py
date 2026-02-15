import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ------------------ SETUP ------------------
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

last_action_time = 0
delay = 1.2

# Smoothing variables
smooth = 6
prev_x, prev_y = 0, 0
clicking = False

# ------------------ FIST DETECTION ------------------
def is_fist(hand):
    wrist = hand.landmark[0]
    tips = [8, 12, 16, 20]

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

        # ---------- HIGHLIGHT ----------
        # ---------- INDEX FINGER (CURSOR CONTROL) ----------
        index_tip = hand.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        screen_x = np.interp(x, [0, w], [0, screen_w])
        screen_y = np.interp(y, [0, h], [0, screen_h])

        # Smooth movement
        curr_x = prev_x + (screen_x - prev_x) / smooth
        curr_y = prev_y + (screen_y - prev_y) / smooth

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        # ---------- PINCH (CLICK) ----------
        thumb_tip = hand.landmark[4]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        distance = np.hypot(tx - x, ty - y)

        if distance < 35:
            if not clicking:
                pyautogui.mouseDown()
                clicking = True
        else:
            if clicking:
                pyautogui.mouseUp()
                clicking = False

        # ---------- PALM POSITION ----------
        palm = hand.landmark[0]
        x_px = int(palm.x * w)
        y_px = int(palm.y * h)

        # ---------- GESTURE CONTROLS ----------
        # ---------- FIST → ZOOM OUT ----------
        if is_fist(hand) and current_time - last_action_time > delay:
            pyautogui.hotkey('ctrl', '-')
            print("ZOOM OUT")
            last_action_time = current_time

         # ---------- PALM UP → ZOOM IN ----------
        elif y_px < h // 3 and current_time - last_action_time > delay:
            pyautogui.hotkey('ctrl', '+')
            print("ZOOM IN")
            last_action_time = current_time
            
         # ---------- LEFT / RIGHT SLIDE ----------
        elif x_px < w // 4 and current_time - last_action_time > delay:
            pyautogui.press('left')
            print("PREVIOUS SLIDE")
            last_action_time = current_time

        elif x_px > 3 * w // 4 and current_time - last_action_time > delay:
            pyautogui.press('right')
            print("NEXT SLIDE")
            last_action_time = current_time

    # Draw guide lines
    cv2.line(frame, (w//4, 0), (w//4, h), (0, 255, 0), 2)
    cv2.line(frame, (3*w//4, 0), (3*w//4, h), (0, 255, 0), 2)
    cv2.line(frame, (0, h//3), (w, h//3), (255, 0, 0), 2)

    cv2.imshow("Smart Presentation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()