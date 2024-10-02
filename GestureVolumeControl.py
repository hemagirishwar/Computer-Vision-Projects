import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

cap=cv2.VideoCapture(0)
mpHand=mp.solutions.hands
hand=mpHand.Hands(min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils
actual_volume=400
ptime=0
Devices=AudioUtilities.GetSpeakers()
Interface=Devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
Volume=cast(Interface,POINTER(IAudioEndpointVolume))
Volrange=Volume.GetVolumeRange()
minVol=Volrange[0]
maxVol=Volrange[1]
volpercentage=0
while True:
    r,f=cap.read()
    f=cv2.flip(f,1)
    img=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    result=hand.process(img)
    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            lmlist=[]
            for ID,lm in enumerate(handloc.landmark):
                h,w,c=f.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([ID,cx,cy])
            x1,y1=lmlist[4][1],lmlist[4][2]
            x2,y2=lmlist[8][1],lmlist[8][2]
            lcx,lcy=(x1+x2)//2,(y1+y2)//2
            cv2.circle(f,(x1,y1),10,(255,0,255),-1)
            cv2.circle(f,(x2,y2),10,(255,0,255),-1)
            cv2.circle(f,(lcx,lcy),10,(255,0,255),-1)
            cv2.line(f,(x1,y1),(x2,y2),(255,0,255),2)
            length=math.hypot(x2-x1,y2-y1)
            vol=np.interp(length,[10,260],[minVol,maxVol])
            actual_volume=int(np.interp(length,[10,260],[400,100]))
            volpercentage=int(np.interp(length,[10,260],[0,100]))
            Volume.SetMasterVolumeLevel(vol,None)
            if length<50:
                cv2.circle(f,(lcx,lcy),10,(0,255,0),-1)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.rectangle(f,(50,100),(80,400),(0,255,0),3)
    cv2.rectangle(f,(50,actual_volume),(80,400),(0,255,0),-1)
    cv2.putText(f,str(int(volpercentage)),(50,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
    cv2.putText(f,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    cv2.imshow("Video",f)
    if cv2.waitKey(1)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()
