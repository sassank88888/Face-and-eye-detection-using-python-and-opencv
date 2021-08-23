
""" @author shashank singh"""



import cv2 as cv

cap=cv.VideoCapture(0,cv.CAP_DSHOW)
face = cv.CascadeClassifier("C:/Users/arpit/Desktop/opencv/Image-Processing-Tutorials-main/Data/cascades/haarcascade_frontalface_default.xml")

eyes=cv.CascadeClassifier("C:/Users/arpit/Desktop/opencv/Image-Processing-Tutorials-main/Data/cascades/haarcascade_eye.xml")


def finder(img):
    gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.putText(img, "Status:", (10, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv.LINE_AA)

    s,k=gray.shape[::-1]

    faces=face.detectMultiScale(gray,1.3,4)

    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

        cv.putText(img,"Face Detected",(70,30),cv.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv.LINE_AA)

        roi_gray=gray[y:y+h,x:x+w]
        roi_colr=img[y:y+h,x:x+w]

        eye=eyes.detectMultiScale(roi_gray,1.3,15)

        for ex,ey,ew,eh in eye:

            cv.rectangle(roi_colr,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
            cv.putText(img, ", Eye Detected", (190, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv.LINE_AA)




    return  img




while True:

    ret,frame=cap.read()
    frame=cv.flip(frame,2)




    cv.imshow("video",finder(frame))


    if cv.waitKey(5) &0xFF==ord("q"):

        break





cv.destroyAllWindows()