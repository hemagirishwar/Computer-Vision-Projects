import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh()
mpDraw=mp.solutions.drawing_utils
mpDrawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    r,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=faceMesh.process(imgRGB)
    if result.multi_face_landmarks:
        for faceloc in result.multi_face_landmarks:
            for ID,lm in enumerate(faceloc.landmark):
                h,w,c=img.shape
                cx,cy=lm.x*w,lm.y*h
                print(ID,cx,cy)
            mpDraw.draw_landmarks(img,faceloc,mpFaceMesh.FACEMESH_CONTOURS,mpDrawSpec,mpDrawSpec)
    cv2.imshow("Video",img)
    if cv2.waitKey(1)==ord("p"):
        break
cap.release()
cv2.destroyAllWindows()
