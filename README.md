# Description

This is my implementation of a pneumonia detection model for X-ray scans of the lungs. It is based on Kaggle's [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview) competition. My implementation takes inspiration from the discussions among the winners of the competition [here](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/discussion?sort=hotness).

# Details

1) To start with the PyTorch implementation of the [Retinanet detection model](https://github.com/yhenon/pytorch-retinanet) was used.

2) The following major modifications were made to the code:
   1) **Training Data Transformations**: The original code had only the random horizontal flip transformation for the input images. Transformations like rotation, translation, scaling, and shear were added. 
   2) **Ensemble Model**: The code was modified to train an ensemble of 3 models using scikit-learn's StratifiedKFold
   3) **Regularization**: To mitigate overfitting in the detection, dropout was added in the classification part of the network.

4) Training details:
   1) Batchsize: 200
   2) Image size: 256
   3) Number of epochs: 13
   4) Optimiser: Adam, LR 1e-5 

6) Inference details:
   1) Classification score threshold (to decide whether an anchor box is a detection): 0.05
   2) Non-maximum suppression (NMS) IOU threshold: 0.1
   3) For ensembling, an IoU threshold of 0.5 was used at which boxes detected by different models were merged (by taking a weighted average of their coordinates). For bounding boxes that did not overlap between the models, a classification threshold value of 0.065 was used to decide whether the solitary box should be retained.
   4) For reasons explained further below, the true detections in the test images were systematically smaller than the ones in the training images. So after computing the detection boxes from the ensemble, every box was resized to 87.5% of its size. 


(put create_csv code, training and prediction code in separate files to be run as separate commands and not in notebook)
## Training

1) Create csv annotation file. Use the command:
   > python create_trainingcsv.py

3) Start training with command:
   > python train.py --csv_file annots_file.csv --csv_classes class_list.csv  --epochs 13 --batchsize 200

The trained models will be saved in the current directory.


## Prediction

1) Create csv annotation file. Use the command:
   > python create_inferencecsv.py

3) Start training with command:
   > python csv_prediction.py --csv_classes class_list.csv --csv_test test_annots.csv --model_epoch 13 --num_images_topredict 3000 --score_threshold 0.05

The csv file containing the predictions will be created in the current directory.


## Visualise

You can visualize the model's output by running: 
   > python visualize.py --csv_classes class_list.csv --csv_val test_annots.csv --model model_final.pt --score_threshold 0.05


## Results

Kaggle provided a test set of 3000 images with the correct outputs hidden. Based on the model's submitted predictions on the test images, Kaggle computes the Average Precision (AP), averaged for IOU thresholds ranging from 0.4 to 0.75. The Mean Average Precision (MAP) for our final ensemble model was <number>. 

This score is not particularly high but neither are the scores of the top scorers of the competition. A major issue is that 

To conclude, more work is required to build a more accurate RAG-based MCQ answering model.

