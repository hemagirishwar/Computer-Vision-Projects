import cv2
import mediapipe as mp
import time

myList=[img1.jpg,img2.jpg,img3.jpg,img4.jpg,img5.jpg,img6.jpg]  # Add the path of the images the iamges are provided in the directory
overlapimages=[]

mpHands=mp.solutions.hands
hand=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
ptime=0
tipIds=[4,8,12,16,20]
for img in myList:
    images=cv2.imread(f'{folderpath}\\{img}')
    images=cv2.resize(images,(200,200))
    overlapimages.append(images)       
cap=cv2.VideoCapture(0)
while True:
    r,f=cap.read()
    f=cv2.flip(f,1)
    imgRGB=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    result=hand.process(imgRGB)
    if result.multi_hand_landmarks:
        for handloc in result.multi_hand_landmarks:
            li=[]
            for ID,lm in enumerate(handloc.landmark):
                h,w,c=f.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                li.append([ID,cx,cy])
            if li:
                finger=[]
                #Thumb
                if li[tipIds[0]][1]<li[tipIds[0]-1][1]:
                        finger.append(1)
                else:
                        finger.append(0)
                #4 Fingers
                for i in range(1,5):
                    if li[tipIds[i]][2]<li[tipIds[i]-2][2]:
                        finger.append(1)
                    else:
                        finger.append(0)
                count_fingers=finger.count(1)
                h,w,c=overlapimages[count_fingers-1].shape
                f[0:h,0:w]=overlapimages[count_fingers-1]
                cv2.rectangle(f,(40,260),(150,400),(0,255,0),-1) 
                cv2.putText(f,str(count_fingers),(50,380),cv2.FONT_HERSHEY_PLAIN,9,(0,0,255),4)
            mpDraw.draw_landmarks(f,handloc,mpHands.HAND_CONNECTIONS)       
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(f,f'FPS:{int(fps)}',(210,80),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Video",f)
    if cv2.waitKey(1)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()    
