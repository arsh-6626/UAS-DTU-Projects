import cv2 as cv
import numpy as np

path = "C:\\Users\\HP\\OneDrive\\Desktop\\uas task\\uas takimages\\uas takimages\\5.png"
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


maskbrown = cv.inRange(imgHSV, lowerb, upperb)
maskgreen = cv.inRange(imgHSV, lowerg, upperg)

def getTriangles(img):
    tlist = []
    imgcopy = img.copy()
    imgcopy[maskbrown>0], imgcopy[maskgreen>0] = (153,153,255), (153,255,153)
    imgtriangles = cv.bitwise_not(cv.add(maskbrown, maskgreen))
    contours,hierarchy = cv.findContours(imgtriangles ,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>100:
            cv.drawContours(imgcopy, cnt, -1, (0, 0, 0), 3)
            peri = cv.arcLength(cnt,True)
            #print(peri)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            objCor = len(approx)
            # print(approx)
            xm = (approx[0][0][0] + approx[1][0][0] + approx[2][0][0])//3
            ym = (approx[0][0][1] + approx[1][0][1] + approx[2][0][1])//3
            cv.circle(imgcopy, (xm,ym), 3, (0,0,0), -1)
            tlist.append((xm,ym))
        else:
            continue
    return imgcopy, tlist

def trianglesinmask(mask):
    num = 0
    mask_inv = cv.bitwise_not(mask)
    contours,hierarchy = cv.findContours(mask_inv ,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    # print(hierarchy)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>10:
            peri = cv.arcLength(cnt,True)
            #print(peri)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            objCor = len(approx)
            if objCor ==3:
                num+=1
    return num

while True:
    imgc, tlist = getTriangles(img)
    print(len(tlist))
    red = 0
    for i in range(len(tlist)):
        centercolor = imgc[tlist[i][0],tlist[i][1]]
        if(centercolor==[0,0,255]):
            red+=1
    blue = len(tlist)-red
    print("red", red)
    print("blue", blue)
    
    n_burnt = trianglesinmask(maskbrown)
    n_safe = trianglesinmask(maskgreen)
    # print(n_burnt)
    cv.imshow("xxx", imgc)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

