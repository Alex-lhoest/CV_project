import numpy as np                     # Numerical algorithms on arrays
import cv2                             # OpenCV
import pandas as pd

def soccer_lines_eval(dilation_parameter):
	global_true_positive = 0
	global_false_positive = 0
	global_true_negative = 0
	global_false_negative = 0
	global_lines_detected = 0

	for file_number in range (1,96):
	    img_link = 'Data/images/soccer_%d.png' %file_number
	    annotation_link = 'Data/annotations/soccer_%d.txt' %file_number
	    
	    data = pd.read_csv(annotation_link, header=None)

	    x1_annot = []
	    y1_annot = []
	    x2_annot = []
	    y2_annot = []

	    img = cv2.imread(img_link, cv2.IMREAD_GRAYSCALE)
	    height, width = img.shape[:2]

	    H=height

	    for _, row in data.iterrows():
	        x1_annot.append(int(row[0]))
	        y1_annot.append(int(row[1]))
	        x2_annot.append(int(row[2]))
	        y2_annot.append(int(row[3]))
	    
	    number_lines = len(row)
	    
	    # Assessment of the performance for the lines detection

	    #edge extraction
	    img = cv2.imread(img_link, cv2.IMREAD_GRAYSCALE)

	    iGausKernelSize = 7
	    imgFilt = cv2.GaussianBlur(img, (iGausKernelSize, iGausKernelSize), 0)
	    _, imgThres = cv2.threshold(imgFilt, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
	    iReducFactor = 2
	    iStart = iReducFactor // 2
	    imgReduc = imgThres[iStart::iReducFactor, iStart::iReducFactor]

	    med = np.median( imgReduc)

	    # apply automatic Canny edge detection using the computed median
	    sigma = 0.3
	    loThreshold = int( max( 0, (1.0 - sigma) * med))
	    hiThreshold = int( min( 255, (1.0 + sigma) * med))

	    edges_canny = cv2.Canny( imgReduc, loThreshold, hiThreshold, apertureSize=3, L2gradient=False)
	    edges_for_lines = cv2.dilate(edges_canny, np.ones((2,2), dtype=np.uint8))
	    edges_dilated = cv2.dilate(edges_canny, np.ones((dilation_parameter,dilation_parameter), dtype=np.uint8))

	    Max_gap = 2
	    Min_length = 75

	    height, width = img.shape[:2]
	    mat = np.zeros((height,width))

	    lines = cv2.HoughLinesP(edges_for_lines,1,np.pi/180,100,np.array([]),Min_length,Max_gap)
	    for l in lines:
	        for x1,y1,x2,y2 in l:
	            cv2.line(mat,(2*x1,2*y1),(2*x2,2*y2),(255,255,255),3) 

	    blackReduc = mat[iStart::iReducFactor, iStart::iReducFactor]
	    blackReduc = blackReduc.astype('uint8')

	    img_and_canny = cv2.bitwise_and(blackReduc, edges_canny)

	    img_and_dilated = cv2.dilate(img_and_canny, np.ones((dilation_parameter,dilation_parameter), dtype=np.uint8))

	    edges_negative_dilated = edges_dilated - img_and_dilated


	    #Computation of the true/false positive/negative pixels

	    mat_annotations1 = np.zeros((height,width))

	    for i in range(len(x1_annot)):
	        cv2.line(mat_annotations1,(int(x1_annot[i]),H-int(y1_annot[i])),(int(x2_annot[i]),H-int(y2_annot[i])),(255, 255, 255),2) 


	    #Comparison between adaptive threshold and annotations

	    blackReduc_annotations = mat_annotations1[iStart::iReducFactor, iStart::iReducFactor]

	    blackReduc_annotations = blackReduc_annotations.astype('uint8') 

	    blackReduc_annotations_invert = (255-blackReduc_annotations)

	    true_positive = cv2.bitwise_and(img_and_dilated, blackReduc_annotations)
	    false_positive = cv2.bitwise_and(img_and_dilated, blackReduc_annotations_invert)
	    true_negative = cv2.bitwise_and(edges_negative_dilated, blackReduc_annotations_invert)
	    false_negative = cv2.bitwise_and(edges_negative_dilated, blackReduc_annotations)

	    num_true_positive = cv2.countNonZero(true_positive)
	    num_false_positive = cv2.countNonZero(false_positive)
	    num_true_negative = cv2.countNonZero(true_negative)
	    num_false_negative = cv2.countNonZero(false_negative)

	    pourcent_true_pos = num_true_positive/(num_true_positive+num_false_negative)
	    pourcent_false_pos = num_false_positive/(num_false_positive+num_true_negative)
	    pourcent_true_neg = num_true_negative/(num_false_positive+num_true_negative)
	    pourcent_false_neg = num_false_negative/(num_true_positive+num_false_negative)


	    #Comparison between Hough and annotations

	    alpha = 0.2 #tolerance parameter for lines evaluation

	    good_lines_detected = 0
	    i=0
	    for ll in lines :
	        mat = np.zeros((height,width))
	        for x11,y11,x22,y22 in ll: 
	            cv2.line(mat,(2*x11,2*y11),(2*x22,2*y22),(255, 255, 255),3)
	            pixels_Hough=cv2.countNonZero(mat)
	            num_pixels_in_common = cv2.countNonZero(cv2.bitwise_and(mat_annotations1, mat))
	            if (num_pixels_in_common > alpha*pixels_Hough):
	                good_lines_detected = good_lines_detected + 1
	    
	    global_true_positive = global_true_positive + pourcent_true_pos
	    global_false_positive = global_false_positive + pourcent_false_pos
	    global_true_negative = global_true_negative + pourcent_true_neg
	    global_false_negative = global_false_negative + pourcent_false_neg
	    global_lines_detected = global_lines_detected + good_lines_detected/len(lines)

	global_true_positive = global_true_positive/95*100
	global_false_positive = global_false_positive/95*100
	global_true_negative = global_true_negative/95*100
	global_false_negative = global_false_negative/95*100
	global_lines_detected = global_lines_detected/95*100

	return global_true_positive, global_false_positive, global_true_negative, global_false_negative, global_lines_detected


def roc_curve_soccer():

	true_pos_vector = []
	false_pos_vector = []

	true_pos_vector.append(0)
	false_pos_vector.append(0)

	for dilation_parameter in range (1,6):
	    true_pos, false_pos,_,_,_ = soccer_lines_eval(dilation_parameter)
	    true_pos_vector.append(true_pos/100)
	    false_pos_vector.append(false_pos/100)

	true_pos_vector.append(1)
	false_pos_vector.append(1)

	return true_pos_vector,false_pos_vector