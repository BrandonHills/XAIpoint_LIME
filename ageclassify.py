import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import skimage
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

def main():
	classifier = AgeClassify()
	ans = classifier.process("/Users/HillOfFlame/NLP_InfoSys/XAIpoint/age-gender-estimation/SmallData/wiki_crop/00/test2.jpg", perturbation=50)
	print(ans)

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


	def bucketize(self, lst, n):
		newBuckets = []
		for i in range(int(len(lst)/n + 1)):
			newBuckets.append(sum(lst[i*n:i*n+n]))
		return newBuckets

	def TenYearRangePredictor(self, imgLst):
		return self.AgePredicatorRange(imgLst, 10)

	def FiveYearRangePredictor(self, imgLst):
		return self.AgePredicatorRange(imgLst, 5)

	def AgePredicatorRange(self, imgLst, bucketSize):
	    
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
		file_path = "/Users/HillOfFlame/NLP_InfoSys/XAIpoint/age-gender-estimation/SmallData/wiki_crop/00/test8.jpg"
		img=cv2.imread(file_path) 

		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def get_downsized_image(self, file_path):
		img_size = 64
		orig = self.get_original_image(file_path)
		
		return cv2.resize(orig, (img_size, img_size), interpolation=cv2.INTER_AREA)



	def process(self, file_path, bucketSize=10, perturbation=1000):

		# Get Downsized Image
		resizedImg = self.get_downsized_image(file_path)

		print("downsized image...")

		# Instantiate the Explainer and Segmenter
		explainer = lime_image.LimeImageExplainer(verbose = False)
		segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

		print("generating explanation...")
		# Generate Explanation from LIME
		explanation = explainer.explain_instance(resizedImg, self.TenYearRangePredictor, top_labels=5, hide_color=0, num_samples=perturbation)

		print("generating model predictions")
		# Generate model predictions
		preds=self.TenYearRangePredictor(np.asarray([resizedImg]))[0]


		print("generating masks")
		# now show masks for each class
		imageMasks=[]
		ageRanges=[]

		for i in explanation.top_labels:
		    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)
		    imageMasks.append(label2rgb(mask,temp, bg_label = 0))
		    ageRanges.append("(" + str(i*bucketSize) + "-" + str((i+1)*bucketSize) + ")")


		return (imageMasks, ageRanges)







if __name__ == __main__:
	main()














		   
		