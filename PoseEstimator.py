import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(r"C:\Users\hemag\OneDrive\Desktop\cv module videos\WhatsApp Video 2024-07-14 at 14.17.21_36a75a4d.mp4")
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils
ctime=0
ptime=0
while True:
    r,img=cap.read()
    if not r:
        break
    img=cv2.resize(img,(800,700))    
    imageRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=pose.process(imageRGB)
    if result.pose_landmarks:
        for ID,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),15,(0,255,255),-1)
        mpDraw.draw_landmarks(img,result.pose_landmarks,mpPose.POSE_CONNECTIONS)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
    cv2.imshow("Video",img)
    if cv2.waitKey(10)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()
