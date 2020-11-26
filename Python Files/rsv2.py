import cv2
import numpy as np
from scipy.stats import itemfreq
import time
def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

count=0
start_time = time.time()
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
       
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 400:
            #cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
           # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


            if len(approx) == 3:
                #cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                detected = frame[y-10:y+h+10,x-10:x+w+10,:]
                #print("Detected => ", detected)
                filename='sign'+str(count)+'.jpeg'
                cv2.imwrite(filename, detected)
                count+=1
                print("Count => ", count)
            elif len(approx) == 4:
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
                #detected = frame[y-10:y+h+10,x-10:x+w+10,:]
                #print("Detected => ", detected)
                #filename='sign'+str(count)+'.jpeg'
                #cv2.imwrite(filename, detected)
                #count+=1
                #print("Count => ", count)
            elif len(approx) == 8:
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.putText(frame, "Octagon", (x, y), font, 1, (0, 0, 0))
                detected = frame[y-10:y+h+10,x-10:x+w+10,:]
                #cv2.rectangle(frame, (x-10,y-10), (x+w+10,y+h+10), (255,0,0), 2)
                #print("Detected => ", detected)
                filename='sign'+str(count)+'.jpeg'
                cv2.imwrite(filename, detected)
                count+=1
                print("Count => ", count)
            #elif 8 < len(approx) < 25:
             #   cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))
    ### add circle detection part here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 37)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)

    if not circles is None:
        circles = np.uint16(np.around(circles))
        #print(circles)
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]

        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            #cv2.rectangle(frame, (i[0]-i[2],i[1]-i[2]), (i[0]+i[2],i[1]+i[2]), (255,0,0), 2)
            
            detected = frame[i[1]-i[2]:i[1]+i[2],i[0]-i[2]:i[0]+i[2],:]
            filename='sign'+str(count)+'.jpeg'
            cv2.imwrite(filename, detected)
            count+=1

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    end_time = time.time()
    if (end_time-start_time)>=5:
        break

cap.release()
cv2.destroyAllWindows()