import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()
mpDraw=mp.solutions.drawing_utils
ctime=0
ptime=0
while True:
    r,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=faceDetection.process(imgRGB)
    if result.detections:
        for ID,detection in enumerate(result.detections):
            # mpDraw.draw_detection(img,detection)
            ih,iw,ic=img.shape                                                             
            fdc=detection.location_data.relative_bounding_box
            fd=int(fdc.xmin*iw),int(fdc.ymin*ih),int(fdc.width*iw),int(fdc.height*ih)
            cv2.rectangle(img,fd,(255,0,255),2)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(20,60),cv2.FONT_ITALIC,1,(255,0,255),2)
    cv2.putText(img,f'{int(detection.score[0]*100)}%',(fd[0],fd[1]-10),cv2.FONT_ITALIC,1,(0,255,255),2)
    cv2.imshow("Video",img)
    if cv2.waitKey(1)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()
