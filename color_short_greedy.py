import cv2 as cv
import numpy as np

lowerg = np.array([34, 0, 0])
upperg = np.array([102, 255, 255])

lowerb = np.array([1, 0, 0])
upperb = np.array([34, 255, 255])

lowerbt = np.array([58, 0, 0])
upperbt = np.array([130, 255, 255])

lowerrt = np.array([0, 0, 0])
upperrt = np.array([0, 255, 255])

houses = []
p_list = []
p_r_list = []

def changecolors(img, mask1, mask2):
    imgcopy = img.copy()
    imgcopy[mask1 > 0], imgcopy[mask2 > 0] = (153, 153, 255), (153, 255, 153)
    return imgcopy

def drawTriangles(img, mask1, mask2):
    imgcopy = img.copy()
    imgmasked = cv.bitwise_not(cv.add(mask1, mask2))
    contours, _ = cv.findContours(imgmasked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        objCor = len(approx)

        if area > 100 and objCor == 3:
            cv.drawContours(imgcopy, cnt, -1, (0, 0, 0), 3)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv.circle(imgcopy, (cX, cY), 5, (0, 0, 0), -1)
    return imgcopy

def getCentroids(img, mask1, mask2):
    imgmasked = cv.bitwise_not(cv.add(mask1, mask2))
    contours, _ = cv.findContours(imgmasked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    tlist = []
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        objCor = len(approx)
        
        if area > 100 and objCor == 3:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                tlist.append((cX, cY))
    return tlist

def addLabels(img, list_of_points, label):
    imgcopy = img.copy()
    for point in list_of_points:
        cv.putText(imgcopy, label, point, cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 1)
    return imgcopy

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def greedy_tsp(points):
    n = len(points)
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        last_point = path[-1]
        next_point = None
        min_dist = float('inf')
        
        for i in range(n):
            if not visited[i]:
                dist = distance(points[last_point], points[i])
                if dist < min_dist:
                    min_dist = dist
                    next_point = i
        
        path.append(next_point)
        visited[next_point] = True

    path.append(0)
    
    return path

for i in range(1, 11):
    img_number = str(i)
    path = f"C:\\Users\\HP\\OneDrive\\Desktop\\uas task\\uas takimages\\uas takimages\\{img_number}.png"
    
    print(path)
    img = cv.imread(path)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    maskbrown = cv.inRange(imgHSV, lowerb, upperb)
    maskgreen = cv.inRange(imgHSV, lowerg, upperg)
    maskred = cv.inRange(imgHSV, lowerrt, upperrt)
    maskblue = cv.inRange(imgHSV, lowerbt, upperbt)

    imgcopy = changecolors(img, maskbrown, maskgreen)
    imgcopy = drawTriangles(imgcopy, maskbrown, maskgreen)
    
    brownred = getCentroids(img, maskbrown, maskred)
    brownblue = getCentroids(img, maskbrown, maskblue)
    greenred = getCentroids(img, maskgreen, maskred)
    greenblue = getCentroids(img, maskgreen, maskblue)
    
    all_points = brownred + brownblue + greenred + greenblue
    tsp_path = greedy_tsp(all_points)

    for i in range(len(tsp_path) - 1):
        cv.line(imgcopy, all_points[tsp_path[i]], all_points[tsp_path[i+1]], (0, 255, 0), 2)

    imgcopy = addLabels(imgcopy, brownred, "Brown&Red")
    imgcopy = addLabels(imgcopy, brownblue, "Brown&Blue")
    imgcopy = addLabels(imgcopy, greenblue, "Green&Blue")
    imgcopy = addLabels(imgcopy, greenred, "Green&Red")

    num_brown_houses = len(brownblue) + len(brownred)
    num_green_houses = len(greenblue) + len(greenred)
    houses.append([num_brown_houses, num_green_houses])

    p_b = 2 * len(brownblue) + len(brownred)
    p_g = 2 * len(greenblue) + len(greenred)
    p_list.append([p_b, p_g])

    print(f"Output {i}")
    print(f"Number of houses on burnt grass = {num_brown_houses}")
    print(f"Number of houses on green grass = {num_green_houses}")
    print(f"Priority of burnt patch = {p_b}")
    print(f"Priority of green patch = {p_g}")
    print(f"Priority ratio = {p_b/p_g}")
    p_r_list.append(p_b / p_g)

    cv.imshow(f"result{i}", imgcopy)
    cv.waitKey()

print()
print("--------------------------------------------------------------------------------------------------------")
print("n_houses =", houses)
print("priority_houses =", p_list)
print("priority ratio =", p_r_list)
