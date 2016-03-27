import math
import time
import cv2
import numpy as np
from copy import copy

eccentricity = 0.6
#acceptable distance(pixels) between cocentric circle centers
distance_threshold = 15
# number of circles needed for a valid target(times 2); 2 circles are often overlayed
min_circles = 5
#pixels: used to identify repeat circles(stacked circles). Problem caused by findContours()
radius_tolerance = 2
#Tolerance used in comparing actaul ratios and preceived ratios
ratio_tolerance = 0.015


#target specific data
#target_code is the unique ratio between rings
target_code_def = np.array([0.8,0.91,0.76,0.84,0.7,0.66,0.49])
target_code = target_code_def
#the outer_ring is a scaling factor for targets of various sizes; radius of outer ring in meters
outer_ring = 0.08255

#define field of view
cam_hfov = 70.42
cam_vfov = 43.3

#define camera size
cam_width = 640
cam_height = 480

cam = cv2.VideoCapture(0)
def current_milli_time():
	return int(time.time() * 1000)

def add_target_highlights(image, target):
	#create a shallow copy of image
	img = copy(image)
	if(len(img.shape) < 3):
		img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	if target is not None:
		for i in range(0,len(target)):
			cv2.ellipse(img,target[i],(0,255,0),2)
	return img

def get_distance_from_pixels(size_in_pixels, actual_size,fov,img_size):
	 # avoid divide by zero by returning 9999.9 meters for zero sized object 
    if (size_in_pixels == 0):
        return 9999.9
    # convert num_pixels to angular size
    return actual_size / pixels_to_angle(size_in_pixels,fov,img_size)

def pixels_to_angle(num_pixels,fov,img_size):
    return num_pixels * math.radians(fov) / img_size

def analyze_frame(img):
	#start timer
	start = current_milli_time()

	#blur image and grayscale
	#img = cv2.medianBlur(img,5)

	#check for a colored image
	if(len(img.shape)>2):
		#grayscale image
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#adaptive threshold
	avg, null, null, null = cv2.mean(img)
	thres = int(avg)
	ret,img = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)

	#dilate
	kernel = np.ones((5,5),np.uint8)
	img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT)

	#canny edge detector
	edges = cv2.Canny(img,100,200,3)

	if edges is not None:

		#locate contours
		contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		#turn contours into ellipses
		circles = np.empty((len(contours)),object)
		circlesCnt = 0
		for i in xrange(0,len(contours)):
			contour = contours[i]
			#make sure contour contains enough point for an ellipse
			if(len(contour) > 4):
				#detect an ellipse
				ellipse = cv2.fitEllipse(contour)
				#only take ellipses which are round
				if checkEccentricity(ellipse,eccentricity):
					circles[circlesCnt] = ellipse
					circlesCnt += 1

			#if circles were found then we look for nested circles
		if circlesCnt > 0:

			#get rid of null elements
			circles = np.resize(circles,circlesCnt)
			#look for nested ellipses
			nestedCircles = detectNested(circles)

			#if at least min_circles circles are nested look for target
			#Times min_circles by two because we haven't removed repeat/stacked circles yet
			if len(nestedCircles) > (min_circles * 2):

				#look for circles with a common center
				finalTarget, center = findCommonCenter(nestedCircles)

				#we found the target position on xy-plane
				if finalTarget is not None:

					#decode the target rings for a list of ring ratios
					ratios = tagAspectRatio(finalTarget)


					#try to calculate distance to target
					if ratios is not None:
						distance = calcDistToTarget(finalTarget,ratios)

						stop = current_milli_time()
						return (stop-start,center, distance, finalTarget)
						#unable to calculate distance due to invalid data
					else:
						stop = current_milli_time()
						return ( stop-start, center, 0, finalTarget)


	#unable to locate target
	stop = current_milli_time()
	return (stop-start,None,0,None)

#distCenters - distance between two ellipses
def distCenters(ellipse1,ellipse2):
	#distance between centers
	distance = math.sqrt(math.pow((ellipse1[0][0]-ellipse2[0][0]),2) + math.pow((ellipse1[0][1] - ellipse2[0][1]),2))
	return distance

#detectNested- return circles which are nested within other circles
def detectNested(rawCircles):
	size = len(rawCircles)
	nestedCircles = np.empty(size, object)
	nestedCnt = 0
	for i in xrange(0,size):
		nested = False
		for j in xrange(i, size):
			if i != j:
				circle1 = rawCircles[i]
				circle2 = rawCircles[j]
				#average major and minor axises
				radius1 = (circle1[1][0] + circle1[1][1]) /2.0
				radius2 = (circle2[1][0] + circle2[1][1]) /2.0

				distance = distCenters(circle1,circle2)

				#check if a circle is nested within another circle
				if(distance < math.fabs(radius1 - radius2)):
					nested = True
		#add the base circle if it is nested
		if nested:
			nestedCircles[nestedCnt] = rawCircles[i]
			nestedCnt += 1
	#remove null objects
	nestedCircles  = np.resize(nestedCircles,nestedCnt)

	return nestedCircles


#checkEccentricity - checks if an ellipse is 'round' enough
def checkEccentricity(ellipse, threshold):
	#threshold = 1 for perfect circles
	if ellipse[1][0] * 1.0/ ellipse[1][1] > threshold:
		return True
	return False

#findCommonCenter - locates a group of circles which share a the most common center. Returns the group and the center point
def findCommonCenter(nestedCircles):

	size = len(nestedCircles)

	#sort by radius
	for i in xrange(0,size):
		baseCircle = nestedCircles[i]
		smallestRadius = (baseCircle[1][0] + baseCircle[1][1]) /2.0
		smallest = i

		for j in xrange(i,size):
			circle = nestedCircles[j]
			radius = (circle[1][0] + circle[1][1]) /2.0
			if(radius < smallestRadius):
				smallestRadius = radius
				smallest = j

		nestedCircles[i] = nestedCircles[smallest]
		nestedCircles[smallest] = baseCircle

	#look at all circles
	#add all circles that are within a certain threshold distance
	#compare circle pairs and see which one has the most circles
	concentricCombos = np.empty([size,size],object)


	#start with the largest circle and scan all smaller circles and see if it is concentric with the large circle
	maxConcentricCnt = 1
	maxConcentricIndex = 0

	#stores circle centers
	xSum = np.zeros(size)
	ySum = np.zeros(size)

	for i in xrange(size-1,0,-1):
		outer = nestedCircles[i]
		concentricCombos[i][0] = outer
		cnt = 1


		for j in xrange(i, 0, -1):
			inner = nestedCircles[j]
			#outer circle and inner circle have same center, are different
			if (distCenters(outer,inner) < distance_threshold) and (i != j):
				#check that the circle isn't a repeat(a problem with findContours)
				previous = concentricCombos[i][cnt -1]
				radPrev = (previous[1][0] + previous[1][1]) /2.0
				radCurr = (inner[1][0] + inner[1][1]) /2.0
				#if the circle is cocentric and unique, add it
				if(radPrev - radCurr) > radius_tolerance:
					concentricCombos[i][cnt] = inner

					xSum[i] += inner[0][0]
					ySum[i] += inner[0][1]

					cnt += 1

		if(cnt > maxConcentricCnt):
			maxConcentricCnt = cnt
			maxConcentricIndex = i

	#no concentric circles
	if(maxConcentricCnt < min_circles):
		return None,None

	#choose the circle set with the most concentric circles
	mostConcentric = concentricCombos[maxConcentricIndex]
	mostConcentric = np.resize(mostConcentric, maxConcentricCnt)

	#calculate meanCenter
	meanCenter = xSum[maxConcentricIndex] / (maxConcentricCnt - 1), ySum[maxConcentricIndex]/(maxConcentricCnt - 1)

	return mostConcentric, meanCenter

#tagAspectRatio- processes the final target and calculates the ratio between rings. returns an array of ratios
def tagAspectRatio(target):
	size = len(target)
	#ratios = np.empty((size-1)*size/2.0, float)
	ratios = np.empty(size-1,float)
	cnt = 0

	for i in xrange(0,size-1):
		circle1 = target[i]
		circle2 = target[i+1]
		radius1 = (circle1[1][0] + circle1[1][1]) /2.0
		radius2 = (circle2[1][0] + circle2[1][1]) /2.0

		ratio = radius2 / radius1
		ratios[cnt] = round(ratio,3)
		cnt += 1
	return ratios

#calculateRingSize - based on ring ID number and target size, calculate the size of a specific ring
def calculateRingSize(ringNumber):
	radius = outer_ring #in meters

	#actualRadius Outer ring size * ratio[n] * ratios[n + 1] ...
	for i in xrange(0,ringNumber):
		radius = radius * target_code[i]

	return radius #in meters

#calcDistToTarget - processes a target and calculates distance to the target
def calcDistToTarget(target, ratios):
	distance = 0
	readings = 0
	for i in xrange(0,len(ratios)):
		ratio = ratios[i]
		for j in xrange(0,len(target_code)):


			if(math.fabs(target_code[j] - ratio) <= ratio_tolerance):
				circle1 = target[i] #outer ring
				circle2 = target[i+1] #inner ring
				radius1 = (circle1[1][0] + circle1[1][1]) /2.0
				radius2 = (circle2[1][0] + circle2[1][1]) /2.0
				fov = math.sqrt(cam_vfov**2 + cam_hfov**2)
				img_size = math.sqrt(cam_width**2 + cam_height**2)


				dist1 = get_distance_from_pixels(radius1, calculateRingSize(j),fov,img_size)
				dist2 = get_distance_from_pixels(radius2, calculateRingSize(j+1),fov,img_size)
				distance += (dist1 + dist2 )/2.0


				readings += 1

	#can not decode target
	if(readings == 0):
		return -1
	#average all distance readings
	return distance/(readings * 1.0)


while(cam.isOpened()):
	ret, img = cam.read()
	results = analyze_frame(img)
	rend_image = add_target_highlights(img, results[3])
	cv2.imshow('gambar',rend_image)
	print ('RunTime: {0} Center: {1} Distance: {2} Raw Target: {3}'.format(results[0],results[1],results[2],results[3]))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#print 'aaa'

#results = analyze_frame(img)
cam.release()
cv2.destroyAllWindows()
#cv2.destroyAllWindows()
