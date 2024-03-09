import cv2
import numpy as np
import time
import math
import mediapipe as mp
import pyautogui


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        b_box = []
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 230), 2, cv2.FILLED)
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            b_box = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (b_box[0] - 15, b_box[1] - 15),
                              (b_box[2] + 15, b_box[3] + 15), (255, 255, 0), 2)
        return self.lm_list, b_box

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, pt1, pt2, img, draw=True):
        x1, y1 = self.lm_list[pt1][1], self.lm_list[pt1][2]
        x2, y2 = self.lm_list[pt2][1], self.lm_list[pt2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 165, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 165, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 240), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
        len_line = math.hypot(x2 - x1, y2 - y1)
        return len_line, img, [x1, y1, x2, y2, cx, cy]


# Main code
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(detection_conf=0.75, max_hands=1)

# Get the size of your screen
screenW, screenH = pyautogui.size()

# Variables to keep track of the drag and click states
dragging = False
clicking = False

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img, draw=True)
    lmList, b_box = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        # Get the coordinates of the index finger tip and thumb tip
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[4][1], lmList[4][2]

        # Invert the X axis
        screenX = np.interp(wCam - x1, (0, wCam), (0, screenW))
        # Keep the Y axis as is
        screenY = np.interp(y1, (0, hCam), (0, screenH))

        # Move the cursor to the position of the index finger tip
        pyautogui.moveTo(screenX, screenY)

        # Calculate the distance between the index finger tip and thumb tip for drag and drop
        drag_distance = math.hypot(x2 - x1, y2 - y1)

        # If the distance is short, simulate a click and hold for dragging
        if drag_distance < 40:
            if not dragging:
                pyautogui.mouseDown()
                dragging = True
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False

        # Check if the index finger is up for clicking
        if lmList[8][2] < lmList[6][2]:  # Compare the y-coordinate of the index fingertip and the PIP joint
            if not clicking:
                pyautogui.click()
                clicking = True
        else:
            clicking = False

    cv2.imshow("Frame", img)
    cv2.waitKey(1)