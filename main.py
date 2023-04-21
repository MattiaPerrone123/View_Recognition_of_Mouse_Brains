from Dataset_Mouse import Dataset_Mouse
from CNN_Model import CNN_Model
from Model_Evaluator import Model_Evaluator
from Utils import *

#Path that are used to import data from the three different datasets
data_dir="/content/gdrive/MyDrive/Colab Notebooks/ML Challenge Genentech/mouse_brain_challenge"

path_horizontal=os.path.join(data_dir,"horizontal")
path_coronal=os.path.join(data_dir,"coronal")
path_sagittal=os.path.join(data_dir,"sagittal")

path_horizontal_CCFv3=os.path.join(data_dir,"CCFv3","horizontal")
path_coronal_CCFv3=os.path.join(data_dir,"CCFv3","coronal")
path_sagittal_CCFv3=os.path.join(data_dir,"CCFv3","sagittal")

path_nissl=os.path.join(data_dir,"nissl","ara_nissl_10.nrrd")

target_size=(512, 512)


#1) 
#Initializing Dataset_Mouse class
brain_dataset=Dataset_Mouse(target_size)

#Loading the data
X_bm, y_bm=brain_dataset.load_data([path_horizontal, path_coronal, path_sagittal])
X_CCFv3, y_CCFv3=brain_dataset.load_data([path_horizontal_CCFv3, path_coronal_CCFv3, path_sagittal_CCFv3])
X_nissl, y_nissl=brain_dataset.load_data([path_nissl])

#Removing black images
X_bm, y_bm=brain_dataset.remove_black_images(X_bm, y_bm)
X_CCFv3, y_CCFv3=brain_dataset.remove_black_images(X_CCFv3, y_CCFv3)
X_nissl, y_nissl=brain_dataset.remove_black_images(X_nissl, y_nissl)

#Splitting and merging dataset
X_train, X_test, y_train, y_test=brain_dataset.split_and_merge_datasets(X_bm, X_CCFv3, X_nissl, y_bm, y_CCFv3, y_nissl)




#2)
#Initializing CNN_Model class
my_model = CNN_Model()

#Performing cross-validation
scores = my_model.fit_cv(X_train, y_train, n_splits=5)

#Training the model
history = my_model.train(X_train, y_train, epochs=10, validation_split=0.2)

#Predicting output classes on the training and test set
y_pred_train=my_model.predict(X_train)
y_pred_test=my_model.predict(X_test)



#3)
#Initializating Model_Evaluator class
my_evaluator=Model_Evaluator(my_model.model, X_test, y_test, y_pred_test)

#Plotting training and validation loss and accuracy
my_evaluator.plot_training_history(history)

#Calculating evaluation metrics
accuracy, f1, roc_auc = my_evaluator.evaluate()

#Defining the classes for the confusion matrix
classes = ['horizontal', 'coronal', 'sagittal']

#Confusion matrix for the training set
my_evaluator=Model_Evaluator(my_model.model, X_train, y_train, y_pred_train)
my_evaluator.plot_confusion_matrix(classes)

#Confusion matrix for the test set
my_evaluator=Model_Evaluator(my_model.model, X_test, y_test, y_pred_test)
my_evaluator.plot_confusion_matrix(classes)

#Plotting the misclassified images
my_evaluator=Model_Evaluator(my_model.model, X_train, y_train, y_pred_train)
misclassified_idx=my_evaluator.plot_misclassified_images()

#Plotting Class activation maps of the misclassified images
if misclassified_idx is not None:
  for i in range(len(misclassified_idx)):
    my_evaluator.plot_gradcam(X_train[misclassified_idx[i]])
else:
  print("No misclassified images")   


#4)
# Repeat points 2) and 3) training the model on only of the three datasets (CCFv3),
# and testing it on another one
X_train, y_train = shuffle(X_CCFv3, y_CCFv3, random_state=15)
X_test, y_test = shuffle(X_bm, y_bm, random_state=15)

