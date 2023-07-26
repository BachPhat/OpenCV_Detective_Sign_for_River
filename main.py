import cv2 as cv
import numpy as np
largest_area=0
n=3
l_trackbar = 255
l_trackbar_max = 255
a_trackbar = 255
a_trackbar_max = 255
b_trackbar = 255
b_trackbar_max = 255
l_H_trackbar = 255
l_H_trackbar_max = 255
a_H_trackbar = 255
a_H_trackbar_max = 255
b_H_trackbar = 255
b_H_trackbar_max = 255

print("Start.....")
#frame= cv.imread('Photo/RIVER0.JPG',1)
cap = cv.VideoCapture(0)
print("GO")
kern = np.ones((5, 5))
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def nothing(x):
    pass

cv.namedWindow("MY CONTROL BAR")
trackbar_name_l = 'L x %d' % l_trackbar_max
cv.createTrackbar(trackbar_name_l, "MY CONTROL BAR", 0, l_trackbar_max, nothing)
trackbar_name_l_H = 'L_H x %d' % l_H_trackbar_max
cv.createTrackbar(trackbar_name_l_H, "MY CONTROL BAR", 0, l_H_trackbar_max, nothing)
trackbar_name_a = 'A x %d' % a_trackbar_max
cv.createTrackbar(trackbar_name_a, "MY CONTROL BAR", 0, a_trackbar_max, nothing)
trackbar_name_a_H = 'A_H x %d' % a_H_trackbar_max
cv.createTrackbar(trackbar_name_a_H, "MY CONTROL BAR", 0, a_H_trackbar_max, nothing)
trackbar_name_b = 'B x %d' % b_trackbar_max
cv.createTrackbar(trackbar_name_b, "MY CONTROL BAR", 0, b_trackbar_max, nothing)
trackbar_name_b_H= 'B_H x %d' % b_H_trackbar_max
cv.createTrackbar(trackbar_name_b_H, "MY CONTROL BAR", 0, b_H_trackbar_max, nothing)

while True:
    cs1 = 0
    cs2 = 0
    cs3 = 0
    cs4 = 0

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    #------------------#
    l_trackbar = int(cv.getTrackbarPos(trackbar_name_l, "MY CONTROL BAR"))
    l = np.array([l_trackbar])
    a_trackbar = int(cv.getTrackbarPos(trackbar_name_a, "MY CONTROL BAR"))
    a = np.array([a_trackbar])
    b_trackbar = int(cv.getTrackbarPos(trackbar_name_b, "MY CONTROL BAR"))
    b = np.array([b_trackbar])
    #--------------------#
    l_H_trackbar = int(cv.getTrackbarPos(trackbar_name_l_H, "MY CONTROL BAR"))
    l_H = np.array([l_H_trackbar])
    a_trackbar = int(cv.getTrackbarPos(trackbar_name_a_H, "MY CONTROL BAR"))
    a_H = np.array([a_H_trackbar])
    b_trackbar = int(cv.getTrackbarPos(trackbar_name_b_H, "MY CONTROL BAR"))
    b_H = np.array([b_H_trackbar])
    Low = np.array([l, a,b])
    High = np.array([l_H, a_H, b_H])
    img_rgb_mask = cv.inRange(img_gray,Low,High)
    img_gray = cv.bitwise_and(img_gray, img_gray, mask=img_rgb_mask)
    cv.imshow('fillter', img_gray)
    # thresholding it to reveal the shapes in the image

    edge =cv.Canny(img_gray, 1000, 2000, apertureSize=5)
    # invert edges to create one big water blob
    contours, hierarchy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, [255, 255, 0], 5)
    Circle=[]
    for contour in contours:
        area = cv.contourArea(contour)
        #approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        #if (len(approx) > 1) :
            #Circle.append(contour)
    #for contour in Circle:
        if area > largest_area:
            largest_area = area
            largest_contour = contour



    point = []
    for contour in contours:
        area = cv.contourArea(contour)
        if (area>(largest_area/1000)) :

            print("area:", area)
            #print("approx",len(approx))
            #if area > (largest_area / 30):  # filter_Contour
            M = cv.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                #print('x:', x, ';y:', y)
                point.append([x, y])
                img_center = cv.circle(frame, (x, y), radius=10, color=(255, 0, 0), thickness=-1)
    # Draw line center
#-----------------Right-left-------------------------#
    print('size', frame.shape)
    print(point)
    size = frame.shape
    sizeX = size[1] // 2
    sizeY = size[0] // 2
    cv.line(frame, (sizeX, sizeY), (sizeX, size[0]), (255, 0, 255), 5)
    cv.line(frame, (sizeX, sizeY), (sizeX, 0), (255, 0, 255), 5)
    print('x:', sizeX, 'y:', sizeY)
    img_center = cv.circle(frame, (sizeX, sizeY), radius=8, color=(0, 255, 255), thickness=-1)
    #-----------------left-------------------#
    left = []
    for l in point:
        if l[0] < sizeX:
            left.append(l)
    print(left)
    a = len(left) - 1
    print(a)
    if len(left) > 1:
        cv.line(frame, (left[0]), (left[a]), (255, 0, 0), 5)
        # ..............................
        if (left[0][0] - left[a][0]) !=0:
            cs1 = (left[0][1] - left[a][1]) / (left[0][0] - left[a][0])
            cs2 = left[0][1] - left[0][0] * cs1
            print(cs1)
            print(cs2)

    #----------------Right------------------#
    right = []
    for r in point:
        if r[0] > sizeX:
            right.append(r)
    print(right)
    b = len(right) - 1
    print(b)
    if len(right)>1:
        cv.line(frame, (right[0]), (right[b]), (255, 0, 0), 5)
        if (right[0][0] - right[b][0]) !=0:
            cs3 = (right[0][1] - right[b][1]) / (right[0][0] - right[b][0])
            cs4 = right[0][1] - right[0][0] * cs3
            print(cs3)
            print(cs4)
    if cs3 !=0 and cs4 !=0 and cs1 !=0 and cs2 !=0:
        # ------------------------------------
        x_boat = int(-(cs2 - cs4) / (cs1 - cs3))
        x_boat = x_boat
        y_boat = x_boat * cs1 + cs2
        print('x:', x_boat)
        # -------------------------------------
        y_boat = int(x_boat * cs1 + cs2)
        print('y;', y_boat)
        cv.line(frame, (x_boat, sizeY), (x_boat, size[0]), (255, 150, 200), 5)
        cv.line(frame, (x_boat, sizeY), (x_boat, 0), (255, 150, 200), 5)
        cv.circle(frame, (x_boat, sizeY), radius=8, color=(100, 0, 255), thickness=-1)

    # find contour in image

    cv.imshow('Cv', frame)
    cv.imshow('frame', edge)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()