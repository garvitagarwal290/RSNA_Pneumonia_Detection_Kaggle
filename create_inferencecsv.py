import pandas as pd
import os


#create csv file for test images
image_names = os.listdir('/home/garvit/Downloads/pneumonia-detection-project/pneumonia_detection_data/test_images/')
image_names.sort()
testimage_names = ['/home/garvit/Downloads/pneumonia-detection-project/pneumonia_detection_data/test_images/'+image_name for image_name in image_names]
test_annots = pd.DataFrame({'image_names':testimage_names, 'x1':[None]*len(testimage_names), 'y1':[None]*len(testimage_names), 'x2':[None]*len(testimage_names), 'y2':[None]*len(testimage_names), 'class':[None]*len(testimage_names)})
test_annots.to_csv('test_annots.csv', header=False, index=False)
