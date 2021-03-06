import os
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from scipy.misc import imresize
import skimage
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
import lime
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import pickle
import math
from collections import OrderedDict
import imutils

def main():
	# For Testing
	
	if 1 == 0:
		classifier = AgeClassify()
		filename = "/Users/HillOfFlame/NLP_InfoSys/XAIpoint/age-gender-estimation/SmallData/wiki_crop/00/test3.jpg"
		ans = classifier.process(filename, perturbation=50)
		maskLst=ans[0]

		print("SHAPE OF MASK LIST", np.asarray(maskLst).shape)
		ans2 = classifier.process_facial_feature(filename, maskLst)
		print(ans2)

	if 1 == 0:
		classifier = AgeClassify()
		filename = "/Users/HillOfFlame/NLP_InfoSys/XAIpoint/age-gender-estimation/SmallData/wiki_crop/00/test7.jpg"
		ans = classifier.process(filename, perturbation=50)
		print(ans)
		pickle.dump(ans, open("pickled/ans3.p", "wb"))

		maskLst = ans[0]
		ans2 = classifier.predict_boundingbox(maskLst, (0,0), (63,63), ans[2])
		pickle.dump(ans2, open("pickled/ans4.p", "wb"))

		ans3 = classifier.get_overlay(filename, maskLst, 22)
		pickle.dump(ans3, open("pickled/ans5.p", "wb"))

		img = classifier.get_original_image(filename)

		pic = classifier.draw_bounding_box(img, (30,30), (50,50))

		facialFeatures = classifier.process_facial_feature(filename, maskLst, ans[2])

		print(classifier.laymans_explanation(facialFeatures, ans[2]))

		pickle.dump((facialFeatures, ans[2]), open("pickled/input7.p", "wb"))


		# print("SHAPEC:", np.asarray(maskLst).shape)

		# classifier.export_image_list(maskLst)
	if 1 == 1:
		input = pickle.load(open("pickled/input10.p", "rb"))

		classifier = AgeClassify()


		print(classifier.laymans_explanation(input[0], input[1]))




class AgeClassify:

	def __init__(self):
	## import libraries and neural net weights
		pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
		modhash = '89f56a39a78454e96379348bddd78c0d'
		weight_file = get_file("weights.18-4.06.hdf5", pretrained_model, cache_subdir="pretrained_models", file_hash=modhash)

		img_size = 64
		depth=16
		width=8
		margin=0.4
		self.model = WideResNet(img_size, depth=depth, k=width)()
		self.model.load_weights(weight_file)


	def laymans_explanation(self, facialDict, predictedAge):
		# Separate into older and younger
		
		younger = []
		older = []
		for feature in facialDict.keys():
			if facialDict[feature] - 1 > predictedAge:
				older.append(feature)
			elif facialDict[feature] + 1 < predictedAge:
				younger.append(feature)

		OYL = len(younger)
		OOL = len(older)

		# Removing redundancies
		if "right cheek" in younger and "left cheek" in younger:
			younger.remove("right cheek")
			younger.remove("left cheek")
			younger.append("cheeks")

		if "right cheek" in older and "left cheek" in older:
			older.remove("right cheek")
			older.remove("left cheek")
			older.append("cheeks")

		if "right eye" in younger and "left eye" in younger:
			younger.remove("right eye")
			younger.remove("left eye")
			younger.append("eyes")

		if "right eye" in older and "left eye" in older:
			older.remove("right eye")
			older.remove("left eye")
			older.append("eyes")

		print("FACIAL DICT:", facialDict)
		print("YOUNGER:", younger)
		# Generate the explanation
		
		if len(younger) > 0 or len(older) > 0:
			explanation = "We have found that your "
			
			if len(younger) > 0:
				if len(younger) > 1:
					for feature in younger[:-1]:
						explanation += feature + ", "
					explanation = explanation[:-2]
					explanation += " and " + younger[-1] + " "
				else:
					explanation += younger[-1] + " "
				
				if OYL > 1:
					explanation += "make you look younger than your predicted age "
				else: 
					explanation += "makes you look younger than your predicted age "

				if len(older) > 0:
					explanation += "and that your "

			if len(older) > 0:
				if len(older) > 1:
					for feature in older[:-1]:
						explanation += feature + ", "
					explanation = explanation[:-2]
					explanation += " and " + older[-1] + " "

				else:
					explanation += older[-1] + " "
				if OOL > 1:
					explanation += "make you look older than your predicted age "
				else:
					explanation += "makes you look older than your predicted age "

			explanation = explanation[:-1] + "."
		
		else:
			explanation = "No features stick out to cause irregularities in your age prediction."

		return explanation


	def process(self, file_path, perturbation=50, rnge=5):
		# Should take file_path and return maskLst, age prediction, and an overlay.

		# Get Downsized Image
		origImg = self.get_original_image(file_path)
		resizedImg = self.get_downsized_image(file_path)

		print("downsized image...")

		# Instantiate the Explainer and Segmenter
		explainer = lime_image.LimeImageExplainer(verbose = False)
		segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

		print("generating explanation...")

		# Generate Explanation from LIME
		explanation = explainer.explain_instance(resizedImg, classifier_fn = self.SingleYearPredictor, top_labels=101, hide_color=0, num_samples=perturbation, segmentation_fn=segmenter)

		print("generating model predictions...")
		# Generate model predictions
		preds=self.SingleYearPredictor(np.asarray([resizedImg]))[0]
		specificAgePrediction = [i for i, j in enumerate(preds) if j == max(preds)][0]


		print("collecting masks...")
		# Collect all the masks from each age. Store in a List.
		maskLst=[]
		for i in range(101):
			temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)
			maskLst.append(mask)

		print("generating age range estimation of bounding box...")
		
		# Generate Age Estimation of the range
		vector=self.AreaAgeEstimatorVector(maskLst, (0,0), (63,63))
		rngeVec=self.AreaAgeEstimatorRange(vector, rnge=rnge)

		# Give the most representative range
		rangeMode = [i for i, j in enumerate(rngeVec) if j == max(rngeVec)][0]
		
		# Generate Tuple representing range
		predictionOfBox = (rangeMode, rangeMode+rnge)

		print("returning answer...")
		# Returns a tuple of representative Image+Mask and age range of box.
		# Example: (IMG, (21, 26))

		print("specificAgePrediction", specificAgePrediction)
		print("rngeVec", np.asarray(rngeVec))
		print("vector", vector)


		# New Addition:  Overlaying Mask onto originalSized Image.
		origDim = origImg.shape

		mask = maskLst[specificAgePrediction]

		reMask = imresize(mask, origDim)

		# Make Mask boolean 2D array
		for i in range(len(reMask)):
			for j in range(len(reMask[0])):
				if reMask[i][j] != 0:
					reMask[i][j] = 1

		grayImg = cv2.cvtColor(origImg, cv2.COLOR_RGB2GRAY)
		overlay = label2rgb(reMask,grayImg, bg_label = 0)

		facialFeatures = self.process_facial_feature(file_path, maskLst, specificAgePrediction)

		laymans = self.laymans_explanation(facialFeatures, specificAgePrediction)

		return (maskLst, overlay, specificAgePrediction, laymans)

	def predict_boundingbox(self, maskLst, c1, c2, age):
		# Predict the age of the image based on the bounding box.

		print("generating age range estimation of bounding box...")
		# Generate Age Estimation of the range
		vector=self.AreaAgeEstimatorVector(maskLst, c1, c2)

		print(vector)
		
		# Generate Tuple representing range
		predictionOfBox = self.weighted_average_of_vector(vector, age)

		return predictionOfBox

	def weighted_average_of_vector(self, vector, age):
		
		denominator = sum(vector)

		for i in range(len(vector)):
			vector[i] = vector[i] * i
		

		if denominator == 0:
			avg = age
		else:
			avg = int(sum(vector) / denominator)

		return avg

	def draw_bounding_box(self, img, c1, c2, colour=(246,207,109), thicc=2): 
		# cv2.imwrite('store/img1.png',img)
		# cv2.imwrite('store/img1.jpg',img)

		imglocal = (255 * img.copy()).astype(np.uint8)


		# Scale the coords
		newC1 = ((int(c1[0]/63.0 * img.shape[0])), int(c1[1]/63.0 * img.shape[1]))

		newC2 = ((int(c2[0]/63.0 * img.shape[0])), int(c2[1]/63.0 * img.shape[1]))

		cv2.rectangle(imglocal, newC1, newC2, colour, thicc)

		# cv2.imwrite('store/img2xx.png',cv2.rectangle(img, newC1, newC2, colour, thicc))
		# cv2.imwrite('store/img2.jpg',img)

		return imglocal

	def get_overlay(self, file_path, maskLst, agePrediction):

		# Get Image
		origImg = self.get_original_image(file_path)

		# Getting the composite mask
		mask = maskLst[agePrediction]

		# New Addition:  Overlaying Mask onto originalSized Image.
		reMask = imresize(mask, origImg.shape)

		# Make reMask boolean 2D array
		for i in range(len(reMask)):
			for j in range(len(reMask[0])):
				if reMask[i][j] != 0:
					reMask[i][j] = 1

		grayImg = cv2.cvtColor(origImg, cv2.COLOR_RGB2GRAY)
		overlay = label2rgb(reMask,grayImg, bg_label = 0)



		return overlay
		
	def composite_masks(self, maskLst, perc=.5):
		numMasks = len(maskLst)
		threshold = int(math.ceil(numMasks*perc))
	    
		x, y = len(maskLst[0]), len(maskLst[0][0])

		# Make empty numpy array
		merged = np.zeros((x,y))
		for n in range(numMasks):
			merged += maskLst[n]
		
		for i in range(x):
			for j in range(y):
				if math.isnan(merged[i][j]): 
					merged[i][j]=0
      
				merged[i][j] = min(1,int(merged[i][j]/threshold))  
		return merged

	def bucketize(self, lst, n):
		newBuckets = []
		for i in range(int(len(lst)/n + 1)):
			newBuckets.append(sum(lst[i*n:i*n+n]))
		return newBuckets

	def TenYearRangePredictor(self, imgLst):
		return self.AgePredictorRange(imgLst, 10)

	def FiveYearRangePredictor(self, imgLst):
		return self.AgePredictorRange(imgLst, 5)
	
	def SingleYearPredictor(self, imgLst):
		return self.AgePredictorRange(imgLst, 1)

	def AgePredictorRange(self, imgLst, bucketSize):
	    
		# Must resize the image to 64 by 64
		img_size = 64
		ResizedImgLst = []
		for img in imgLst:
			resized = cv2.resize(img, (img_size, img_size))
			ResizedImgLst.append(resized)
		
		# Turn into Numpy Array
		ResizedImgLst = np.asarray(ResizedImgLst)
		
		# Predict using pretrained model
		predLst = self.model.predict(ResizedImgLst)[1]
		
		results = []
		for pred in predLst:
			results.append(np.asarray(self.bucketize(pred, bucketSize)))

		return results

	def get_original_image(self, file_path):
		# Import Image
		img=cv2.imread(file_path) 

		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def get_downsized_image(self, file_path):
		img_size = 64
		orig = self.get_original_image(file_path)
		
		return cv2.resize(orig, (img_size, img_size), interpolation=cv2.INTER_AREA)

		# Create a function that will count the number of pixels in a bounding box for a certain mask
	def countPixels(self, mask, c1,c2):

		count = 0
		(x1, y1), (x2, y2) = c1, c2
		for x in range(x1, x2 + 1):
			for y in range(y1,y2 + 1):
				if not math.isnan(mask[x][y]):
					count += int(mask[x][y])

		return count

	# Given two coordinates in the 64x64 matrix, return age estimation.
	def AreaAgeEstimatorVector(self, maskLst, c1, c2):
		votes=[]
		for i in range(len(maskLst)):
			votes.append(self.countPixels(maskLst[i], c1,c2))
		return votes

	# Given a vote-vector, calculate the 5-year range with the most votes
	def AreaAgeEstimatorRange(self, vector, rnge=5):
		rangeVector = []
		for i in range(len(vector) - rnge + 1):
			rangeVector.append(sum(vector[i:i+rnge]))
		return rangeVector

	def shape_to_numpy_array(self, shape, dtype="int"):
		# initialize the list of (x, y)-coordinates
		coordinates = np.zeros((68, 2), dtype=dtype)

		# loop over the 68 facial landmarks and convert them
		# to a 2-tuple of (x, y)-coordinates
		for i in range(0, 68):
		    coordinates[i] = (shape.part(i).x, shape.part(i).y)

		# return the list of (x, y)-coordinates
		return coordinates

	def bounding_box_for_points(self, ptLst):
		x_coor, y_coor = [], []
		for i in range(len(ptLst)):
			x_coor.append(ptLst[i][0])
			y_coor.append(ptLst[i][1])
		c1 = (min(x_coor), min(y_coor))
		c2 = (max(x_coor), max(y_coor))
		return (c1, c2)

	def get_facial_landmarks(self, file_path):
		## Return the shape 
		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		print("FILE:", file_path)

		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(file_path)
		image = imutils.resize(image, width=64)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale image

		shape = predictor(gray, detector(gray, 1)[0])
		shape = self.shape_to_numpy_array(shape)

		FACIAL_LANDMARKS_BBox = OrderedDict([
			("mouth", self.bounding_box_for_points(shape[48:68])),
			("right eye", self.bounding_box_for_points(shape[36:42])),
			("left eye", self.bounding_box_for_points(shape[42:48])),
			("nose", self.bounding_box_for_points(shape[27:35])),
			("right cheek", self.bounding_box_for_points([shape[2], shape[7]])),
			("left cheek", self.bounding_box_for_points([shape[11], shape[16]]))])	

		return FACIAL_LANDMARKS_BBox

	def resize_maskLst(self, maskLst, dims):
		newLst = []
		for mask in maskLst:
			# New Addition:  Overlaying Mask onto originalSized Image.
			reMask = imresize(mask, dims)

			# Make reMask boolean 2D array
			for i in range(len(reMask)):
				for j in range(len(reMask[0])):
					if reMask[i][j] != 0:
						reMask[i][j] = 1

			newLst.append(reMask)
		return newLst

	def process_facial_feature(self, file_path, maskLst, age):
		# For each facial feature, tally up the votes for the feature region
		# and average them out to determine the region's age estimation.

		est_dict = dict()
		FACIAL_LANDMARKS_BBox = self.get_facial_landmarks(file_path)
		for landmark in FACIAL_LANDMARKS_BBox.keys():
			c1, c2 = FACIAL_LANDMARKS_BBox[landmark]
			est_dict[landmark] = self.predict_boundingbox(maskLst, c1, c2, age)
		return est_dict


if __name__ == "__main__":
	main()














		   
		