import pandas as pd
import os

# create training and validation annotations csv file
anns = pd.read_csv('/home/garvit/Downloads/pneumonia-detection-project/pneumonia_detection_data/train_labels.csv')
train_annots = anns.copy() 

train_annots['width'] = train_annots['x'] + train_annots['width']
train_annots['height'] = train_annots['y'] + train_annots['height']

train_annots.loc[train_annots['Target']==0, 'Class'] = ''
train_annots.loc[train_annots['Target']==1,'Class'] = 'Lung_Opacity'

train_annots['patientId'] = ['/home/garvit/Downloads/pneumonia-detection-project/pneumonia_detection_data/train_images/{}.jpg'.format(name) for name in train_annots['patientId']]

train_annots.drop(columns=['Target'], inplace=True)
train_annots.rename(columns={"patientId": "Image Path", "x": "x1", "y": "y1", "width": "x2", "height": "y2"}, inplace=True)

train_annots['x1'] = pd.to_numeric(train_annots['x1'], errors='coerce').astype('Int64')
train_annots['x2'] = pd.to_numeric(train_annots['x2'], errors='coerce').astype('Int64')
train_annots['y1'] = pd.to_numeric(train_annots['y1'], errors='coerce').astype('Int64')
train_annots['y2'] = pd.to_numeric(train_annots['y2'], errors='coerce').astype('Int64')

train_annots.to_csv('annots_file.csv', header=False, index=False)

#create csv file for object classes
classes = pd.DataFrame({"class_name":["Lung_Opacity"], "id":[0]})
classes.to_csv('class_list.csv', header=False, index=False)
