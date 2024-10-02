import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
ptime=0
ctime=0
while True:
    r,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            for ID,lm in enumerate(handloc.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(ID,cx,cy)
                if ID==4:
                    cv2.circle(img,(cx,cy),25,(0,255,255),-1)
        mpDraw.draw_landmarks(img,handloc,mpHands.HAND_CONNECTIONS)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("Video",img)
    if cv2.waitKey(1)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()
