import cv2 as cv
import numpy as np
import math

imgT = cv.imread('t.jpg')
imgT = cv.resize(imgT, (360,428))
# imgT = cv.resize(imgT, (640,480))

cap = cv.VideoCapture(0)

def matchshape(bestcnt, cntlist):
    minn = math.inf
    # _, imgTthresh2 = cv.threshold(cv.cvtColor(imgT2, cv.COLOR_BGR2GRAY),127,255,cv.THRESH_BINARY)
    GrayT = cv.cvtColor(imgT, cv.COLOR_BGR2GRAY)
    cannyT = cv.Canny(GrayT, 150,200)
    contours1, hierarchy = cv.findContours(cannyT, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours1) == 0:
        print("No contours found in sample.")
        return None
    # cv.drawContours(imgContour, contours1, -1, (255, 0, 0), 3)
    cnt1=contours1[0]
    bestcunt = None
    # print(cnt1)
    # cnt2, _  = cv.findContours(imgTthresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    # cv.drawContours(imgT,cnt1, -1, (0,0,255),3)
    for cnt in cntlist:
        min2 = cv.matchShapes(cnt, cnt1, 1, 0.0)
        if min2<0.5:
            if min2<=minn:
                minn = min2
                bestcunt = cnt
    if bestcunt is not None:
        cv.drawContours(imgContour, [bestcunt], -1, (0, 255, 0), 3)
    return bestcunt


def angle_box(img, cnt):
    if cnt is not None:
        # angle_deg = cv.minAreaRect()
        rect_angle = cv.minAreaRect(cnt)
        print(rect_angle)
        box = cv.boxPoints(rect_angle)
        box = np.int0(box)
        x = int(rect_angle[0][0])
        y = int(rect_angle[0][0])
        cv.drawContours(img,[box],0,(0,0,255),2)
        cv.putText(img, f"Angle: {str(int(rect_angle[2]))} deg", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# _, imgTthresh = cv.threshold(cv.cvtColor(imgT, cv.COLOR_BGR2GRAY),127,255,cv.THRESH_BINARY)
# cnt1, _  = cv.findContours(imgTthresh, cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# cv.drawContours(imgT, cnt1, -1, (255, 0, 0), 3)

def anglefinder(cnt, center):
    angle = -1
    if cnt is not None:
        
        a=cnt.ravel()[0]
        b=cnt.ravel()[1]
        angle=str(int(np.degrees(np.arctan((center[1]-b)/(center[0]-a)))))
        return angle
    

while True:

    _, frame = cap.read()
    img = frame
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgContour = img.copy()
    blur = cv.bilateralFilter(gray_frame,9,75,75)
    thresh = cv.Canny(blur,300,350)
    kernel = np.ones((5,5),np.uint8)
    dilla = cv.dilate(thresh, kernel)
    bestcunt = None
    cuntlist, _ = cv.findContours(dilla, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    # print(matchshape(bestcunt, cuntlist))
    # cv.drawContours(imgContour, cuntlist, -1, (0,0,255),2)
    Tcontour = [matchshape(bestcunt, cuntlist)]
    cv.drawContours(imgContour, Tcontour, -1, (255, 0, 0), 3)
    angle_box(imgContour, Tcontour[0])
    cv.imshow("cunt", imgContour)
    # cv.imshow("T", dilla)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break