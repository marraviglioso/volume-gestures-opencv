import math
import cv2
import time
import alsaaudio
import numpy as np
import GestureTrackingController as gtc

##
xCam, yCam = 640, 480
##

cap = cv2.VideoCapture(0)
cap.set(3, xCam)
cap.set(4, yCam)
pTime = 0

m = alsaaudio.Mixer()
minVol = 0
maxVol = 100
detector = gtc.handDetector(detectionCon=0.7)

while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        spread = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(spread, [35, 280], [minVol, maxVol])
        print((int(vol)))
        m.setvolume(int(vol))

        if spread <= 50:
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 5)
    cv2.imshow("Debugging window", frame)
    cv2.waitKey(1)
