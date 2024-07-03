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


#assert torch.__version__.split('.')[0] == '1'

use_gpu = True
cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(cuda))


def draw_caption(image, box, caption):

	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')
	
	parser.add_argument('--score_threshold',help='Classification score threshold for a detection',type=str, default='0.05')

	parser = parser.parse_args(args)

	dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([v2.Normalize(mean=[0.49, 0.49, 0.49], std=[0.229, 0.229, 0.229]), v2.Resize(256, antialias=True)]))

	sampler_val = AspectRatioBasedSampler(dataset_val, [i for i in range(len(dataset_val))], batch_size=1, drop_last=False, randomise=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	if use_gpu and torch.cuda.is_available():
		retinanet = torch.load(parser.model)
		retinanet = retinanet.cuda()
	else:
		retinanet = torch.load(parser.model, map_location=torch.device('cpu'))

	if use_gpu and torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()

	#unnormalize = UnNormalizer()

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			if use_gpu and torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float(), float(parser.score_threshold))
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float(), float(parser.score_threshold))
			
			idxs = np.where(scores.cpu()> float(parser.score_threshold))
			
			unnormalize = v2.Compose([v2.Normalize(mean=[0,0,0], std=[1/0.229, 1/0.224, 1/0.225]),
                                v2.Normalize(mean = [-0.485, -0.456, -0.406], std = [ 1., 1., 1. ])])
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			#print(dataset_val.image_names[idx].split('/')[-1])
			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				#print(x1,y1,x2,y2, label_name)
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

			plt.imshow(img, interpolation = 'bicubic')
			plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
			plt.show()


if __name__ == '__main__':
 main()
