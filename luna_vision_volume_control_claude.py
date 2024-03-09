import cv2
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp


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
pTime = 0
min_dist = 25
max_dist = 190
vol = 0
vol_bar = 340
vol_perc = 0
area = 0
vol_color = (250, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(detection_conf=0.75, max_hands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Volume Range -65 - 0
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img, draw=True)
    lmList, b_box = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        # Filter based on Size
        area = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1]) // 100
        if 200 < area < 1000:

            # Find Dist btw index & thumb
            len_line, img, line_info = detector.findDistance(4, 8, img)

            # Convert Vol
            vol_bar = np.interp(len_line, [min_dist, max_dist], [340, 140])
            vol_perc = np.interp(len_line, [min_dist, max_dist], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 10
            vol_perc = smoothness * round(vol_perc / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevel(vol_range[0] + (vol_perc / 100) * (vol_range[1] - vol_range[0]), None)
                cv2.circle(img, (line_info[4], line_info[5]), 5, (255, 255, 0), cv2.FILLED)
                vol_color = (135, 0, 255)
            else:
                vol_color = (135, 0, 255)

            # Min - Max Vol Button Color
            if len_line < min_dist:
                cv2.circle(img, (line_info[4], line_info[5]), 5, (0, 0, 255), cv2.FILLED)
            elif len_line > max_dist:
                cv2.circle(img, (line_info[4], line_info[5]), 5, (0, 255, 0), cv2.FILLED)

    # Drawings
    cv2.rectangle(img, (55, 140), (85, 340), (255, 255, 0), 3)
    cv2.rectangle(img, (55, int(vol_bar)), (85, 340), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f'Vol = {int(vol_perc)} %', (18, 380), cv2.FONT_HERSHEY_COMPLEX, 0.6, (51, 255, 255), 2)
    curr_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol set to: {int(curr_vol)} %', (410, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, vol_color, 2)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Frame", img)
    cv2.waitKey(1)
