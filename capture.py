import cv2

vdo=cv2.VideoCapture(0)
model=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
i=1
while True:
    reg,img= vdo.read()
    if reg==False:
        break
    cv2.putText(img,'press c for quit',(10,20),cv2.FONT_HERSHEY_PLAIN,2,(255,150,200))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=model.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(90,90))

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255),2)
        face=img[y:y+h,x:x+w]
        cv2.imwrite(f'image3/img{i}.png',face)
        i+=1
    cv2.imshow('vdo',img)
    key=cv2.waitKey(25)
    if key==ord('c'):
        break

cv2.destroyAllWindows()
vdo.release()