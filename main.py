import cv2 as cv
from deepface import DeepFace


def croppingImages(frames, x,y,w,h):
    res = frames[y:y+h,x:x+w]
    return res

#taking in the data from the camera in realtime 
cam = cv.VideoCapture(0) 
facecascade = cv.CascadeClassifier("faces_data.xml")

while(True): 
    sucess, frame = cam.read() #capturing the frames fro mthe camera 
    g_f = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #converting the frame matrix into  binary to reduce the resource requirements 
    faces = facecascade.detectMultiScale(g_f,1.1,6)

    count = 0
    data = []
    for (x,y,w,h) in faces: 
        cv.rectangle(frame,(x,y),(x+w,y+h),(225,0,225),5)
        cv.putText(frame,"Person",(x,y),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),3)
        data = croppingImages(frame,x,y,w,h)

    try:
        print(DeepFace.analyze(data))
    except:
        None
        
    cv.imshow("Output",frame)
    if cv.waitKey(1) == ord("c"):
        break

cam.release()
cv.destroyAllWindows()