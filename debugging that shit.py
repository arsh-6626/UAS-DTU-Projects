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

def isBrown(tri):
    color = maskbrown[tri[0],tri[1]+16]
    if color > 0:
        return True
    else:
        return False
        
def isGreen(tri):
    color = maskgreen[tri[0],tri[1]+16]
    # cv.putText(imgcopy, str(color), tri, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
    if color > 0:
        return True
    else:
        return False

def isRed(tri):
    color = maskred[tri[0], tri[1]+1]
    if color > 0:
        return True
    else:
        return False

def isBlue(tri):
    color = maskblue[tri[0], tri[1]+1]
    if color > 0:
        return True
    else:
        return False
    
while True:
    imgcopy = img.copy()
    testcase = (564, 207)
    cv.circle(imgcopy, testcase, 5, (255, 255, 0), -1)
    print("Green", isGreen(testcase), "Brown", isBrown(testcase), "Red", isRed(testcase), "Blue", isBlue(testcase))
    print(maskgreen[testcase], maskbrown[testcase], maskred[testcase], maskblue[testcase])

    cv.imshow("img", imgcopy)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv.Canny(imgBlur,100,100)
    # cv.imshow("green", maskgreen)
    # cv.imshow("brown", maskbrown)
    # cv.imshow("blue", maskblue)
    cv.imshow("red", imgCanny)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break