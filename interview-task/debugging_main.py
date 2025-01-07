import cv2 as cv
import numpy as np


path = "C:\\Users\\HP\\OneDrive\\Desktop\\uas task\\uas takimages\\uas takimages\\2.png"
img = cv.imread(path)
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)


#green grass:
lowerg = np.array([34, 0, 0])
upperg = np.array([102,255,255])

#brown grass:
lowerb = np.array([1,0,0])
upperb = np.array([34,255,255])

#blue triangle
lowerbt = np.array([58,0,0])
upperbt = np.array([130,255,255])

#red triangle
lowerrt = np.array([0,0,0])
upperrt = np.array([0,255,255])

#declaration of masks
maskbrown = cv.inRange(imgHSV, lowerb, upperb)
maskgreen = cv.inRange(imgHSV, lowerg, upperg)
maskred = cv.inRange(imgHSV, lowerrt, upperrt)
maskblue = cv.inRange(imgHSV, lowerbt, upperbt)

def drawTriangles(img, mask1, mask2):
    #mask1- grass where we want the object to be
    #mask2 -the triangles which we want to eliminate
    #e.g mask1 = brownmask and mask2 = redmask, will show all the blue triangles in the brown grass region

    imgcopy = img.copy()
    imgmasked = cv.bitwise_not(cv.add(mask1,mask2))
    cv.imshow("added mask", cv.add(mask1,mask2))
    cv.imshow("inversion", imgmasked)

    #the most interesting part of my code imo, it combines both of non required triangle and fills them up in the required grass region upon inversion the empty patches in the given region turn into triangle contours
    contours,hierarchy = cv.findContours(imgmasked ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt,True)
        #print(peri)
        approx = cv.approxPolyDP(cnt,0.02*peri,True)
        #applying poly approx DP to find all the coordiantes
        objCor = len(approx)
        #checking for 3 sided and sufficient area contours only - in our bitwise not step we also had a mask of the opposite coloured grass, this eliminates that patch from checking
        if area>100 and objCor ==3:
            cv.drawContours(imgcopy, cnt, -1, (0, 0, 0), 3)
            #drawing contours around the required triangles
            M = cv.moments(cnt)
            #finding the centroid via cv.moments()
            if M["m00"]!=0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv.circle(imgcopy, (cX, cY), 5, (0, 0, 0), -1)
            else:
                continue
    return imgcopy

def getCentroids(img, mask1, mask2):
    #mask1- grass where we want the object to be
    #mask2 -the triangles which we want to eliminate
    #this function works the same way as above but just returning every centroid to a list rather than drawing it, appears neat imo to have 2 functions rather than one doing everything
    imgcopy = img.copy()
    tlist = []
    imgmasked = cv.bitwise_not(cv.add(mask1,mask2))
    contours,hierarchy = cv.findContours(imgmasked ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt,True)
        #print(peri)
        approx = cv.approxPolyDP(cnt,0.02*peri,True)
        objCor = len(approx)
        if area>100 and objCor ==3:
            M = cv.moments(cnt)
            if M["m00"]!=0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                tlist.append((cX,cY))
            else:
                continue
    return tlist

def addLabels(img, list, label):
    #just finalising the image by labelling every triangle
    imgcopy = img.copy()
    for ptr in list:
        cv.putText(imgcopy, label, ptr, cv.FONT_HERSHEY_COMPLEX, 0.75, (0,0,0), 1)
    return imgcopy
    
while True:
    imgcopy = img.copy()
    imgcopy = drawTriangles(imgcopy, maskbrown, maskred)

    cv.imshow("img", imgcopy)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv.Canny(imgBlur,100,100)
    # cv.imshow("green", maskgreen)
    # cv.imshow("brown", maskbrown)
    # cv.imshow("blue", maskblue)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
