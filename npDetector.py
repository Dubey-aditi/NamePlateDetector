import cv2

npCascade = cv2.CascadeClassifier("nemePlateDetector/haarcascade_russian_plate_number.xml")
minarea = 500
count=0
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 50)

while True:
    success, img=cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    numberPlates = npCascade.detectMultiScale(imgGray, 1.1, 4)

    for(x, y, w, h) in numberPlates:
        area = w*h
        if area > minarea:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Number plate", (x, y-5), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0), 2)
            npregion = img[y:y+h, x:x+w]
            cv2.imshow("Region of interest", npregion)
    
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite("nemePlateDetector/scanned/NoPlate_"+str(count)+".jpg", npregion)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow("saved", img)
        cv2.waitKey(800)
        count +=1
    elif cv2.waitKey(1) & 0xFF==ord('q'):
        break
