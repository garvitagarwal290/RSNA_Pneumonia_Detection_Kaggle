import argparse

import numpy as np

import torch
from torchvision.transforms import v2

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer

from sklearn.model_selection import KFold, StratifiedKFold
import json


def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_file', help='Path to file containing all annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser = parser.parse_args(args)
	    

	dataset = CSVDataset(train_file=parser.csv_file, class_list=parser.csv_classes, \
		                           transform=v2.Compose([v2.Normalize(mean=[0.49, 0.49, 0.49], std=[0.229, 0.229, 0.229]), \
		                           v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=6, translate=(0.1,0.1), scale=(0.9,1.1), shear=(-3,3,-3,3)), v2.Resize(256, antialias=True)]))

	print("No of images: {}".format(len(dataset)))

	torch.manual_seed(42)
	nsplits = 13
	kfold = StratifiedKFold(n_splits=nsplits, shuffle=True)

	N_val = int(len(dataset)/nsplits)
	N_train = N_val*(nsplits - 1)

	train_ids_list = np.zeros((nsplits, N_train), dtype=int)
	val_ids_list = np.zeros((nsplits, N_val), dtype=int)
	for fold, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(len(dataset)), dataset.get_all_targetlabels())):
		train_ids_list[fold]= train_ids[:N_train]
		val_ids_list[fold] =  val_ids[:N_val]
			
		
	f = open('folds_train.npy', 'wb')
	np.save(f, train_ids_list)
	
	f = open('folds_val.npy', 'wb')
	np.save(f, val_ids_list)
		
if __name__ == '__main__':
    main()
