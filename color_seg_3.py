import cv2 as cv
import numpy as np

#------------------------------------------RANGING OF VALUES-------------------------------------------------------------------
#here all the values of the required colours are ranged under hsv_value ranges that define upper and lower limit of the pixel value for that colour

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

#declaration of empty lists for final output

houses = [] #b,s
p_list = [] #b,s
p_r_list = []

#the function creates different coloured overlays for the brown and green grass

def changecolors(img, mask1, mask2):
    imgcopy = img.copy()
    #using imgcopy just in case we need to check the original img again
    imgcopy[mask1>0], imgcopy[mask2>0] = (153,153,255), (153,255,153)
    # here we have changed the pixel value of every corresponding point in the image for which mask has a value > 0 {note that our created masks have only two values defined i.e 0 and 255}
    return imgcopy


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

for i in range(1,11):
    path = "C:\\Users\\HP\\OneDrive\\Desktop\\uas task\\uas takimages\\uas takimages\\"
    path = path + str(i) + ".png"
    #iteration through all pictures
    print(path)
    img = cv.imread(path)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #creation of mask by the ranged values we found above
    maskbrown = cv.inRange(imgHSV, lowerb, upperb)
    maskgreen = cv.inRange(imgHSV, lowerg, upperg)
    maskred = cv.inRange(imgHSV, lowerrt, upperrt)
    maskblue = cv.inRange(imgHSV, lowerbt, upperbt)

    #applying all our functions 
    imgcopy = changecolors(img, maskbrown, maskgreen)
    imgcopy = drawTriangles(imgcopy, maskbrown, maskgreen)
    brownred = getCentroids(img, maskbrown, maskblue)
    brownblue = getCentroids(img, maskbrown, maskred)
    greenred = getCentroids(img, maskgreen, maskblue)
    greenblue = getCentroids(img, maskgreen, maskred)
    imgcopy = addLabels(imgcopy, brownred, "Brown&Red")
    imgcopy = addLabels(imgcopy, brownblue, "Brown&Blue")
    imgcopy = addLabels(imgcopy, greenblue, "Green&Blue")
    imgcopy = addLabels(imgcopy, greenred, "Green&Red")

    print("\t OUTPUT", i)
    print("Number of houses on burnt grass = ", len(brownblue)+len(brownred))#total no of houses on brown grass
    print("Number of houses on Green Grass = ", len(greenred)+len(greenblue))#total no of houses on brown grass
    houses.append([len(brownblue)+len(brownred),len(greenred)+len(greenblue)])
    p_b = 2*(len(brownblue))+len(brownred)#priority calculation
    p_g = 2*(len(greenblue))+len(greenred)
    p_list.append([p_b,p_g])
    print("Priority of burnt patch = ", p_b)
    print("Priority of green patch = ", p_g)
    print("Priority Ratio = " , p_b/p_g)
    p_r_list.append(p_b/p_g)
    cv.imshow(("result"+str(i)), imgcopy)
    cv.waitKey()

print()
print("--------------------------------------------------------------------------------------------------------")
print("n_houses = ", houses)
print("priority_houses = ", p_list)
print("priority ratio = ", p_r_list)

#i achieved the image sorting problem via creating a dicitonary and then sorting that by value
dict = {}
for i in range(len(p_r_list)):
    dict[("image "+str(i+1))] = p_r_list[i]

keys = list(dict.keys())
values = list(dict.values())
sorted_value_index = np.argsort(values)
sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

sorted_images = list(sorted_dict.keys())
print("images_by_rescue_ratio = ",sorted_images[::-1])

print("---------------------------------------------------------------------------------------------------------")