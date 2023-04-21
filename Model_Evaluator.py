class Model_Evaluator:
    def __init__(self, model, X_true, y_true, y_pred):
        self.model=model
        self.X_true=X_true
        self.y_true=y_true
        self.y_pred=y_pred
    
    """
    This function:
    - returns performance metrics (accuracy, f1-score, roc_auc)
    """
    def evaluate(self):
        y_pred_classes=np.argmax(self.y_pred, axis=1)
        y_true_classes=np.argmax(self.y_true, axis=1)
        accuracy=accuracy_score(y_true_classes, y_pred_classes)
        f1=f1_score(y_true_classes, y_pred_classes, average='weighted')
        roc_auc=roc_auc_score(self.y_true, self.y_pred, multi_class='ovr')

        return accuracy, f1, roc_auc

    """
    This function:
    - plots the confusion matrix
    """
    def plot_confusion_matrix(self, classes):

        y_true=np.argmax(self.y_true, axis=1)
        y_pred=np.argmax(self.y_pred, axis=1)

        cm=confusion_matrix(y_true, y_pred)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt='d'
        thresh=cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i,format(cm[i,j],fmt),
                     horizontalalignment="center",
                     color="white" if cm[i,j]>thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    """
    This function:
    - plots training and validation loss and accuracy
    """
    def plot_training_history(self, history):
        loss=history.history['loss']
        accuracy=history.history['accuracy']
        val_loss=history.history['val_loss']
        val_accuracy=history.history['val_accuracy']

        fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(loss)
        ax1.plot(val_loss)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(['Training', 'Validation'])

        ax2.plot(accuracy)
        ax2.plot(val_accuracy)
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend(['Training', 'Validation'])

        plt.show()

    """
    This function:
    - plots the images misclassified by the model
    - return the indexes of the misclassified images or None (if there are no
      misclassified images)
    """
    def plot_misclassified_images(self):
        y_true=np.argmax(self.y_true, axis=1)
        y_pred=np.argmax(self.y_pred, axis=1)
        misclassified_indices=np.where(y_true != y_pred)[0]

        if len(misclassified_indices)==0:
            print("No misclassified images")
            return

        for i in range(len(misclassified_indices)):
            idx=misclassified_indices[i]
            plt.imshow(self.X_true[idx])
            plt.title(f"True: {y_true[idx]}\nPredicted: {y_pred[idx]}")
            plt.show()

        return misclassified_indices
    
    """
    This function:
     - plots the CAM of the image received in input
    """
    def plot_gradcam(self, img_input):
      x=np.expand_dims(img_input, axis=0)

      predictions=self.model.predict(x)
      predicted_class=np.argmax(predictions[0])

      explainer=GradCAM()
      saliency_map=explainer.explain((x, None), self.model, class_index=predicted_class)

      heatmap=cv2.applyColorMap(np.uint8(255*saliency_map), cv2.COLORMAP_JET)
      overlayed_img=cv2.addWeighted(cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

      plt.title("GradCAM")
      plt.imshow(overlayed_img, cmap='jet')
      plt.imshow(heatmap, cmap='jet')
      plt.colorbar(label='Saliency Score')
      plt.show()
  
    """
    This function:
    - receives in input an array of images and randomly plots the CAM of one of the
      images in the array, as well as the image itself
    - it is just used for visualization purposes
    """
    def plot_random_images_gradcam(self, image_folders):
     for image_folder in image_folders:
        img_file=random.choice(image_folder)
        self.plot_gradcam(img_file)
        plt.title("Image")
        plt.imshow(img_file)



