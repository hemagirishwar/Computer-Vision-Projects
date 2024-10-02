import cv2
import mediapipe as mp
import time
import os
import numpy as np

mpHands=mp.solutions.hands
hand=mpHands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)
mpDraw=mp.solutions.drawing_utils
folderpath=r"Download the images and provide the pathof the images the images are provided in the directory"
pathlist=os.listdir(folderpath)
imgcanvas=np.zeros((720,1280,3),np.uint8)
overlayimages=[]
for path in pathlist:
    img=cv2.imread(f'{folderpath}\\{path}')
    overlayimages.append(img)  
header=overlayimages[0]
drawcolor=(0,0,0)
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
tipid=[4,8,12,16,20]
xc,yc=0,0
while True:
    r,f=cap.read()
    f=cv2.flip(f,1)
    imgRGB=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    f[:125,:1280]=header
    result=hand.process(imgRGB)
    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            li=[]
            for ID,lm in enumerate(handloc.landmark):
                h,w,c=f.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                li.append([ID,cx,cy])
            # tip of index finger
            x1,y1=li[8][1:]    
            # tip of middle finger
            x2,y2=li[12][1:]  
            fingers=[]
            if li[tipid[0]][1]<li[tipid[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for i in range(1,5):
                if li[tipid[i]][2]<li[tipid[i]-1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)        
            if fingers[1] and fingers[2]:
                xc,yc=0,0
                if y1<125:
                    if 179<x1<406:
                        header=overlayimages[1]
                        drawcolor=(255,0,255)
                    elif 551<x1<680:
                        header=overlayimages[2]
                        drawcolor=(255,0,0)
                    elif 852<x1<977: 
                        header=overlayimages[3]
                        drawcolor=(0,255,0)
                    elif 1126<x1<1261:
                        header=overlayimages[4] 
                        drawcolor=(0,0,0)     
                cv2.rectangle(f,(x1,y1-25),(x2,y2+25),drawcolor,-1)           
            if fingers[1] and fingers[2]==False:
                cv2.circle(f,(x1,y1),20,drawcolor,-1)    
                if xc==0 and yc==0:
                    xc,yc=x1,y1
                cv2.line(imgcanvas,(xc,yc),(x1,y1),drawcolor,30)
                xc,yc=x1,y1
            imgGray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
            _,th=cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
            imgInv=cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)
            ba=cv2.bitwise_and(f,imgInv)
            f=cv2.bitwise_or(ba,imgcanvas)    
            mpDraw.draw_landmarks(f,handloc,mpHands.HAND_CONNECTIONS)
    cv2.imshow("Video",f)
    if cv2.waitKey(1)==ord("p"):
        break
cv2.destroyAllWindows()    
cap.release()
