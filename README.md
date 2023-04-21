# View-Recognition-of-Mouse-Brains
The aim of the current project is to implement a Convolutional Neural Network for the recognition of different views of mouse brain images (horizontal, coronal or sagittal).

State-of-the-art EfficientNet B0 has been used for this task. The pre-trained model has been imported from Keras applications, and it has been fine tuned by training just the last five layers. Transfer learning on three different datasets has been used to train the model. As a result, all the images were correctly classified by the model. 

To gain insights into the model's performance generalizability, it was trained on a single dataset and then tested on the other two. Class activation maps were then used to have a better understanding of the model’s decision-making classification process and to identify areas of improvement.	



## Dependencies
A list of the packages used for this project is included in the file Requirements.txt


## Dataset 
Three different datasets, which are the BrainMap Atlas (1036), the Nissl-stained dataset (1312) and CCFv3 dataset (3049), were merged and used to train the model. By giving in input pictures coming from different image modalities and stains, the model becomes more robust for its predictions.

## Images
This folder includes a random image from each of the three planes with its own class activation map


## Models 
State-of-the-art EfficientNet B0 has been used for the recognition of different views of mouse brain images. The pre-trained model (on ImageNet) has been imported from Keras applications, and it has been fine tuned by training just the last five layers for 10 epochs. A global average pooling layer, a dense layer and a softmax layer has been added (the first two also trained), to extract the most relevant features from the observations and to convert the output into a 2D tensor. Then cross-validation has been performed, obtaining an accuracy that can be rounded up to 1 for each of the five folds


## Results and Discussion 
When using the dataset mentioned above, the model is able to classify every image correctly. However, the dataset used has been previously preprocessed, deleting all the images with an average intensity lower than a certain threshold (set arbitrarily to 0.6). If this preprocessing step is not done, the F1-score on the test set is 0.98.

The model was then trained just on one dataset (CCFv3) and tested on another dataset (BrainMap), to evaluate its performance on unseen data. As expected, the model showed signs of overfitting on the training set (F1-score on the training set: 1.00; F1-score on the test set: 0.55). To understand which image features were used by the model for classification in the two scenarios, CAM (Class Activation Maps) were analyzed. By comparing the CAMs of the model trained on the entire dataset and only on CCFv3, it was observed that they were almost identical in both cases. This indicates that in the last case the model was able to understand the relevant features for prediction, but failed to classify the images correctly, because the test samples were too different from the training samples, possibly due to a different population.
