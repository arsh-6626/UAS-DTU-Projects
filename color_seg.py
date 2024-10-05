import cv2 as cv
import numpy as np

path = "C:\\Users\\HP\\OneDrive\\Desktop\\uas task\\uas takimages\\uas takimages\\5.png"
img = cv.imread(path)
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#found hsv ranges for all colours, by using color coder file - took from a tutorial i used while learning
#i think BGR could have also been used for masking but i used HSV as the grass is of different shades - so imo this is better, bgr may also work
#array = [h,s,v]

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



def getTriangles(img):
    tlist = []
    imgcopy = img.copy()
    imgcopy[maskbrown>0], imgcopy[maskgreen>0] = (153,153,255), (153,255,153)
    imgtriangles = cv.bitwise_not(cv.add(maskbrown, maskgreen))
    contours,hierarchy = cv.findContours(imgtriangles ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>100:
            cv.drawContours(imgcopy, cnt, -1, (0, 0, 0), 3)
            M = cv.moments(cnt)
            if M["m00"]!=0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv.circle(imgcopy, (cX, cY), 5, (0, 0, 0), -1)
                cv.circle(imgcopy, (cX, cY+20), 5, (255, 255, 255), -1)
                tlist.append((cX,cY))
            else:
                continue
    return imgcopy, tlist

def isBrown(mask, tri):
    color = mask[tri[0],tri[1]+20]
    if color > 0:
        return True
    else:
        return False
        
def isGreen(mask, tri):
    color = mask[tri[0],tri[1]+20]
    # cv.putText(imgcopy, str(color), tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
    if color > 0:
        return True
    else:
        return False

def isRed(mask, tri):
    color = mask[tri]
    if color > 0:
        return True
    else:
        return False

def isBlue(mask, tri):
    color = mask[tri]
    if color > 0:
        return True
    else:
        return False

#creating seperate masks for both grass patches 
while True:
    kernel = np.ones((5,5),np.uint8)
    maskbrown = cv.inRange(imgHSV, lowerb, upperb)
    _, maskbrown = cv.threshold(maskbrown,0,255,cv.THRESH_BINARY)
    maskbrown = cv.morphologyEx(maskbrown, cv.MORPH_OPEN, kernel)
    maskgreen = cv.inRange(imgHSV, lowerg, upperg)
    _, maskgreen = cv.threshold(maskgreen,0,255,cv.THRESH_BINARY)
    maskgreen = cv.morphologyEx(maskgreen, cv.MORPH_OPEN, kernel)
    maskred = cv.inRange(imgHSV, lowerrt, upperrt)
    _, maskred = cv.threshold(maskred,0,255,cv.THRESH_BINARY)
    maskblue = cv.inRange(imgHSV, lowerbt, upperbt)
    _, maskblue = cv.threshold(maskblue,0,255,cv.THRESH_BINARY)
    imgc, tlist = getTriangles(img)
    redbrown = []
    redgreen = []
    bluebrown = []
    bluegreen = []
    # testcase = (259,363)
    # cv.circle(imgc, testcase, 5, (255, 255, 0), -1)
    # print("Green", isGreen(testcase), "Brown", isBrown(testcase), "Red", isRed(testcase), "Blue", isBlue(testcase))
    # print(maskgreen[testcase], maskbrown[testcase], maskred[testcase], maskblue[testcase])
    # print(tlist)
    for tri in tlist:
        print(tri)
        # testcase = tri
        # print("Green", isGreen(testcase), "Brown", isBrown(testcase), "Red", isRed(testcase), "Blue", isBlue(testcase))
        # print(maskgreen[testcase], maskbrown[testcase], maskred[testcase], maskblue[testcase])
        if isBrown(maskbrown, tri):
            cv.putText(imgc, "brown", (tri[0],tri[1]+10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            # if isRed(tri):
            #     redbrown.append(tri)
            #     cv.putText(imgcopy, "red", tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            # if isBlue(tri):
            #     bluebrown.append(tri)
            #     cv.putText(imgcopy, "blue", tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
           
        elif isGreen(maskgreen, tri):
            cv.putText(imgc, "green", (tri[0],tri[1]+10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
            if isBlue(tri):
                cv.putText(imgc, "blue", tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                bluegreen.append(tri)
            if isRed(tri):
                cv.putText(imgc, "red", tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                redgreen.append(tri)
            
    print("RB", len(redbrown))
    print("RG", len(redgreen))
    print("BB", len(bluebrown))
    print("BG" ,len(bluegreen))
    print("-------------------------------------------------------------------------")
    cv.imshow("img", imgc)
    cv.imshow("green", maskgreen)
    cv.imshow("brown", maskbrown)
    # cv.imshow("blue", maskblue)
    # cv.imshow("red", maskred)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break