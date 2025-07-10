import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

clicking = False
click_start = 0
dragging = False

def get_distance(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            h, w, _ = frame.shape

            x_index = int(lm[8].x * w)
            y_index = int(lm[8].y * h)
            x_thumb = int(lm[4].x * w)
            y_thumb = int(lm[4].y * h)
            x_pinky = int(lm[20].x * w)
            y_pinky = int(lm[20].y * h)

            screen_x = np.interp(x_index, (0, w), (0, screen_width))
            screen_y = np.interp(y_index, (0, h), (0, screen_height))

            pyautogui.moveTo(screen_x, screen_y)

            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            dist_thumb_index = get_distance(x_index, y_index, x_thumb, y_thumb)
            dist_thumb_pinky = get_distance(x_thumb, y_thumb, x_pinky, y_pinky)

            # Drag & Drop
            if dist_thumb_index < 40:
                if not dragging:
                    dragging = True
                    pyautogui.mouseDown()
            else:
                if dragging:
                    dragging = False
                    pyautogui.mouseUp()

            # Left Click (tap)
            if dist_thumb_index < 40 and not clicking:
                click_start = time.time()
                clicking = True
            elif dist_thumb_index < 40 and clicking:
                if time.time() - click_start > 0.5:
                    pyautogui.doubleClick()
                    clicking = False
            elif dist_thumb_index >= 40 and clicking:
                if time.time() - click_start <= 0.5:
                    pyautogui.click()
                clicking = False

            # Right Click
            if dist_thumb_pinky < 40:
                pyautogui.rightClick()
                time.sleep(0.5)  # prevent spam clicks

            # Visual helpers
            cv2.circle(frame, (x_index, y_index), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x_pinky, y_pinky), 10, (0, 0, 255), -1)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

