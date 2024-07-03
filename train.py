import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
#from torchvision import transforms
from torchvision.transforms import v2

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

import time

from sklearn.model_selection import KFold, StratifiedKFold


print('CUDA available: {}'.format(torch.cuda.is_available()))

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--csv_file', help='Path to file containing all annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=1)

	parser.add_argument('--batchsize', help='batchsize for both training and validation', type=int, default=32)

	parser.add_argument('--score_threshold',help='Classification score threshold for a detection',type=str, default='0.05')

	parser.add_argument('--folds_path', type=str, default='')

	parser.add_argument('--fold_no',help='which fold to train the model on',type=int, default='1')

	parser = parser.parse_args(args)


	if parser.csv_file is None:
		raise ValueError('Must provide --csv_file when training,')
	    
	if parser.csv_classes is None:
		raise ValueError('Must provide --csv_classes when training')
	    
	dataset = CSVDataset(train_file=parser.csv_file, class_list=parser.csv_classes, \
		                   transform=v2.Compose([v2.Normalize(mean=[0.49, 0.49, 0.49], std=[0.229, 0.229, 0.229]), \
		                   v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=6, translate=(0.1,0.1), scale=(0.9,1.1), shear=(-3,3,-3,3)), v2.Resize(256, antialias=True)]))


	f = open(parser.folds_path+'folds_train.npy', 'rb')
	train_ids = np.load(f)[parser.fold_no - 1]
	f = open(parser.folds_path+'folds_val.npy', 'rb')
	val_ids = np.load(f)[parser.fold_no -1]
	
	print('Total on {}/{} images'.format(len(train_ids), len(dataset)))
	print('Batch Size: {}'.format(parser.batchsize))


	print('Training on Fold: {}'.format(parser.fold_no))
        
	sampler_train = AspectRatioBasedSampler(dataset, train_ids, batch_size=parser.batchsize, drop_last=False)
	dataloader_train = DataLoader(dataset, num_workers=3, collate_fn=collater, batch_sampler=sampler_train)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	#if(fold==0): print(count_parameters(retinanet))

	use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	if torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)


	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
	#optimizer = optim.SGD(retinanet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

	loss_hist = collections.deque(maxlen=500)

	for epoch_num in range(parser.epochs):

		st = time.time()

		epoch_loss = []
		epoch_clsloss = 0.0
		epoch_regloss = 0.0

		retinanet.train()
		retinanet.module.freeze_bn()

		for iter_num, data in enumerate(dataloader_train):
			optimizer.zero_grad()
			try:
				data['annot'] = torch.cat((data['annot'], data['label']), 2)

				if torch.cuda.is_available():
					classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
				else:
					classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()
				loss = classification_loss + regression_loss

				epoch_clsloss += classification_loss
				epoch_regloss += regression_loss

				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		epoch_clsloss /= (iter_num+1)
		epoch_regloss /= (iter_num+1)        
		print('\nEpoch: {} | Avg Classification loss: {:.5f} | Avg Regression loss: {:.5f} | Avg total loss: {:.5f}'.format(epoch_num+1, float(epoch_clsloss), float(epoch_regloss), float(epoch_clsloss+epoch_regloss)))

		retinanet.eval()
		mAP = csv_eval.evaluate(dataset, val_ids, retinanet, score_threshold=float(parser.score_threshold))

		scheduler.step(np.mean(epoch_loss))

		torch.save(retinanet.module, 'retinanet_fold{}_epoch{}.pt'.format(parser.fold_no, epoch_num+1))

		print('\nElapsed epoch time: {} minutes'.format((time.time()-st)/60))

		retinanet.eval()

        #torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
