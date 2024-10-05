import cv2
import mediapipe as mp
import time
import numpy as np


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

img = cv2.imread(r"Paste the path of any image")
img = cv2.resize(img, (400, 400))

parts = [
    img[:100, 200:300],img[:100, 100:200],img[:100, :100],img[:100, 300:],
    img[100:200, :100],img[100:200, 200:300],img[100:200, 100:200],img[300:, 200:300],
    img[200:300, 100:200],img[200:300, :100],img[200:300, 200:300],img[200:300, 300:],
    img[300:, :100],img[300:, 300:],img[300:, 100:200],img[100:200, 300:]
]
def stack_image(parts):
    return np.vstack([
        np.hstack(parts[:4]),
        np.hstack(parts[4:8]),
        np.hstack(parts[8:12]),
        np.hstack(parts[12:])
    ])

stacked_img = stack_image(parts)

selected_parts = []

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, f = cap.read()
    f = cv2.flip(f, 1)
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f[0:400,880:]=img
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            li = []
            for id, lm in enumerate(handloc.landmark):
                h, w, c = f.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                li.append([id, cx, cy])
            if li:
                finger = []
                x1, y1 = li[8][1], li[8][2]  # Index finger tip
                x2, y2 = li[12][1], li[12][2]  # Middle finger tip

                if li[8][2] < li[7][2]: 
                    finger.append(1)
                else:
                    finger.append(0)
                
                if li[12][2] < li[11][2]:  
                    finger.append(1)
                else:
                    finger.append(0)

                if finger[0] and finger[1]:
                    cv2.rectangle(f,(x1,y1),(x2,y2),(255,255,0),-1)
                elif finger[0] and not finger[1]:
                    cv2.circle(f, (x1, y1), 15, (255, 0, 255), -1)
                    for i in range(4):
                        for j in range(4):
                            x_start, y_start = j * 100, i * 100
                            x_end, y_end = x_start + 100, y_start + 100
                            if x_start < x1 < x_end and y_start < y1 < y_end:
                                selected_parts.append((i, j))
                                if len(selected_parts) == 2:
                                    i1, j1 = selected_parts[0]
                                    i2, j2 = selected_parts[1]
                                    parts[i1 * 4 + j1], parts[i2 * 4 + j2] = parts[i2 * 4 + j2], parts[i1 * 4 + j1]
                                    stacked_img = stack_image(parts)
                                    selected_parts = []                    

    f[:400, :400] = stacked_img

    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(f, handloc, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Img", f)
    
    if cv2.waitKey(1) == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()
