import numpy as np
import cv2 as cv


def drawTriangles(img, mask1, mask2):
    #mask1- grass where we want the object to be
    #mask2 -the triangles which we want to eliminate
    #e.g mask1 = brownmask and mask2 = redmask, will show all the blue triangles in the brown grass region

    imgcopy = img.copy()
    imgmasked = cv.bitwise_not(cv.add(mask1,mask2))

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

#the function creates different coloured overlays for the brown and green grass
def changecolors(img, mask1, mask2):
    imgcopy = img.copy()
    #using imgcopy just in case we need to check the original img again
    imgcopy[mask1>0], imgcopy[mask2>0] = (153,153,255), (153,255,153)
    # here we have changed the pixel value of every corresponding point in the image for which mask has a value > 0 {note that our created masks have only two values defined i.e 0 and 255}
    return imgcopy