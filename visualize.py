import numpy as np
import argparse

import cv2

import torch

import matplotlib.pyplot as plt

import skimage.io
import skimage.transform
import skimage.color
import skimage



use_gpu = True
cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(cuda))


def draw_caption(image, box, caption):

	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	#parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_predictions', help='Path to file containing predictions annotations ', type=str)
	parser.add_argument('--images_path', help='Path to the test images ', type=str)

	parser = parser.parse_args(args)


	lines = open('{}'.format(parser.csv_predictions), 'r').readlines()[1:]

	for line in lines:
	
		split = line.split(',')
		image_name = split[0]
		
		if(split[1]==''): all_boxes = np.zeros(0)
		else:
			annots = split[1].split(' ')
			
			num_boxes = (len(annots)-1)//5
			all_boxes = np.zeros((num_boxes, 4))
			
			for i in range(num_boxes):
				all_boxes[i,0] = float(annots[i*5+1])
				all_boxes[i,1] = float(annots[i*5+2])
				all_boxes[i,2] = all_boxes[i,0] +float(annots[i*5+3])
				all_boxes[i,3] = all_boxes[i,1]+ float(annots[i*5+4])
				
		img = skimage.io.imread('{}{}.jpg'.format(parser.images_path, image_name))
		img[img<0] = 0
		img[img>255] = 255

		img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

		print(image_name)
		for j in range(all_boxes.shape[0]):
			bbox = all_boxes[j]
			x1 = int(round(bbox[0]))
			y1 = int(round(bbox[1]))
			x2 = int(round(bbox[2]))
			y2 = int(round(bbox[3]))
			label_name = 'lung_opacity'
			
			#print(x1,y1,x2,y2, label_name)
			draw_caption(img, (x1, y1, x2, y2), label_name)

			cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

		plt.imshow(img, interpolation = 'bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()


if __name__ == '__main__':
 main()
