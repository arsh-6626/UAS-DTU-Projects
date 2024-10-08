import cv2 as cv
import numpy as np
from route_planner import *
from contour_test import *

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
    cities = brownred + brownblue + greenred + greenblue  # Combine all centroids
    if cities:  # Ensure the list of combination is not empty
        tour = nearest_neighbor_algorithm(cities)
        total_distance = calculate_total_distance(tour, cities)
        print("TSP Tour Order:", tour)
        print("Total TSP Distance:", total_distance)
        # Draw lines
        for j in range(len(tour) - 1):
            pt1 = cities[tour[j]]  # Current city
            pt2 = cities[tour[j + 1]]  # Next city
            cv.line(imgcopy, pt1, pt2, (255, 0, 130), 2) 
        for j in range(len(tour) - 1):
            pt1 = cities[tour[j]]  # Current city
            cv.putText(imgcopy, str(j), pt1, cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,255,255), 2) 
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
print("sorted ratio dictionary = ", sorted_dict)
print("images_by_rescue_ratio = ",sorted_images[::-1])

print("---------------------------------------------------------------------------------------------------------")