
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import v2

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
	
import matplotlib.pyplot as plt
import time
import pandas as pd

from ensemble import GeneralEnsemble

#assert torch.__version__.split('.')[0] == '1'

use_gpu = True
cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(cuda))


def main(args=None):
	parser = argparse.ArgumentParser(description='for getting predictions of bounding boxes on test images')

	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_test', help='Path to file containing test annotations file')

	#parser.add_argument('--model', help='Path to model (.pt) file.')
	
	parser.add_argument('--num_images_topredict', help='Path to model (.pt) file.', type=int, default=20)
	
	parser.add_argument('--score_threshold',help='Classification score threshold for a detection',type=str, default='0.05')
	
	parser.add_argument('--model_epoch',help='Which epoch\'s models to take for ensemble',type=int, default='4')
	
	parser = parser.parse_args(args)

	dataset_test = CSVDataset(parser.csv_test, class_list=parser.csv_classes, transform=transforms.Compose([v2.Normalize(mean=[0.49, 0.49, 0.49], std=[0.229, 0.229, 0.229]),  v2.Resize(256, antialias=True)]))

	sampler_test = AspectRatioBasedSampler(dataset_test, [i for i in range(len(dataset_test))], batch_size=1, drop_last=False, randomise=False)
	dataloader_test = DataLoader(dataset_test, num_workers=1, collate_fn=collater, batch_sampler=sampler_test)

	retinanets = []
	for i in range(3):
		if use_gpu and torch.cuda.is_available():
			retinanet = torch.load('retinanet_fold{}_epoch{}.pt'.format(i+1, parser.model_epoch))
			retinanet = retinanet.cuda()
		else:
			retinanet = torch.load('retinanet_fold{}_epoch{}.pt'.format(i+1, parser.model_epoch), map_location=torch.device('cpu'))

		if use_gpu and torch.cuda.is_available():
			retinanet = torch.nn.DataParallel(retinanet).cuda()
		else:
			retinanet = torch.nn.DataParallel(retinanet)
			
		retinanets.append(retinanet)
			
	all_boxes = []
	all_scores = []
	all_labels = []
			
	for idx, data in enumerate(dataloader_test):
			
		if(idx == parser.num_images_topredict): break
		print('{}/{}'.format(idx+1, parser.num_images_topredict), end='\r')
		
			
		all_detections = []
		
		for i in range(3):
			retinanets[i].eval()
			
			all_detections.append([])

			with torch.no_grad():
				if use_gpu and torch.cuda.is_available():
					scores, labels, boxes = retinanets[i](data['img'].cuda().float(), float(parser.score_threshold))
				else:
					scores, labels, boxes = retinanets[i](data['img'].float(), float(parser.score_threshold))
				
				scores = scores.cpu().numpy()
				labels = labels.cpu().numpy()
				boxes  = boxes.cpu().numpy()

				# correct boxes for image scale
				boxes *= (1024/data['img'].shape[-1])

				# select indices which have a score above the threshold
				indices = np.where(scores > float(parser.score_threshold))[0]
				
				if indices.shape[0] > 0:
					# select those scores
					scores = scores[indices]

					# find the order with which to sort the scores
					scores_sort = np.argsort(-scores)

					for j in range(len(scores_sort)):
						all_detections[i].append(list(boxes[indices[scores_sort[j]]]))
						all_detections[i][j].append(labels[indices[scores_sort[j]]])
						all_detections[i][j].append(scores[scores_sort[j]])
			
		if(sum([len(model_detections) for model_detections in all_detections])==0):
			all_boxes.append(np.array([]))
			all_scores.append(np.array([]))
			all_labels.append(np.array([]))

		else:
			final_boxes, final_scores, final_labels = GeneralEnsemble(all_detections, iou_thresh = 0.5, weights=None)
			all_boxes.append(final_boxes)
			all_scores.append(final_scores)
			all_labels.append(final_labels)
			
					
	anns = pd.read_csv(parser.csv_test, names=['image_names', 'x1', 'y1', 'x2', 'y2', 'class'])
	patientId = anns['image_names'][:parser.num_images_topredict]
	predictionString = []
	
	for i in range(len(patientId)):
		patientId[i] = patientId[i].split('/')[-1].split('.')[0]

		string = ''
		for j in range(len(all_boxes[i])):
			#string+='{} {} {} {} {} '.format(all_scores[i][j], all_boxes[i][j][0], all_boxes[i][j][1], all_boxes[i][j][2]-all_boxes[i][j][0], all_boxes[i][j][3]-all_boxes[i][j][1])  
			#string+='{} {} {} {} {} '.format(all_scores[i][j], all_boxes[i][j][0], all_boxes[i][j][1], (all_boxes[i][j][2]-all_boxes[i][j][0])*0.875, (all_boxes[i][j][3]-all_boxes[i][j][1])*0.875)

			x1 = all_boxes[i][j][0]
			y1 = all_boxes[i][j][1]
			x2 = all_boxes[i][j][2]
			y2 = all_boxes[i][j][3]
			center = (x1*.5+x2*.5, y1*.5+y2*.5)
			newx1 = center[0] - 0.875*(x2-x1)*.5
			newy1 = center[1] - 0.875*(y2-y1)*.5
			neww = 0.875*(x2-x1)
			newh = 0.875*(y2-y1)
			
			string+='{} {} {} {} {} '.format(all_scores[i][j], newx1, newy1, neww, newh)

		predictionString.append(string)

	predictions_dict = {'patientId':patientId, 'PredictionString':predictionString}      
	
	predictions_df = pd.DataFrame(predictions_dict)
	predictions_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
 main()
