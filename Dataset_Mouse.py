class Dataset_Mouse:
    def __init__(self, target_size):
        self.target_size = target_size
    

    def load_images_from_dirs(self, dirs):
      """
      This function: 
      - loads images from a specific path
      - resizes them to a target size 
      - appends the image to a vector, that is returned
      """
      images=[]
      for dir_path in dirs:
        for file_name in os.listdir(dir_path):
            image=cv2.imread(os.path.join(dir_path, file_name))
            resized_image=cv2.resize(image, self.target_size)
            resized_image=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            images.append(resized_image)
      return np.array(images)
    
    
    
    def load_nrrd_image(self, path):
      """
      This function:
      - loads nrrd files from a specific path, importing images
      - scales pixel values of the images imported in the range of (0,255)
      - resizes the image to a target size 
      - converts images from grayscale to RGB
      - appends the image to a vector, that is returned
      """
      data, _=nrrd.read(path)
      images=[]
      for i in range(len(data)):
        image_min=np.min(data[i])
        image_max=np.max(data[i])
        img_rescaled=(data[i]-image_min)/(image_max-image_min)*255.0
        img_fin=np.round(img_rescaled).astype(np.uint8)
        resized_image=cv2.resize(img_fin, self.target_size)
        resized_image=cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        images.append(resized_image)
      return np.array(images)

   
    def load_data(self, paths):
      """
      #This function:
      - calls the two previous functions (load_images_from_dirs and load_nrrd_image)
      - assumes that the images in the nrrd file are coronal views (since this
        is the case for the current dataset)
      - returns images (X) contained in the path folders given in input, 
        as well as their labels (y) 
      - when calling this function, please make sure that nissl data are the last one
        to be imported (as in the current notebook)
      """
      X=[]
      y=[]
      class_label=0
      for path in paths:
        if path.endswith(".nrrd"):
            img=self.load_nrrd_image(path)
            class_label=1
        else:
            img=self.load_images_from_dirs([path])
        X.append(img)
        y.append(np.full(len(img), class_label))
        class_label+=1
      X=np.concatenate(X)
      y=np.concatenate(y)
      y=to_categorical(y, num_classes=3)
      return X, y   


    
    def remove_black_images(self, X, y):
      """
      This function:
      - removes all the images totally black or almost totally black (the choice of
        a threshold equal to 0.6 was made looking directly at the images).
      - These images are deleted both because the model would not be able to classify 
        them correctly and they do not improve learning
      """
      avg_intensity=np.mean(X, axis=(1, 2, 3))
      idx_nonblack=np.where(avg_intensity > 0.6)[0]
      X_nonblack=X[idx_nonblack]
      y_nonblack=y[idx_nonblack]

      return X_nonblack, y_nonblack   
        
    
    def split_and_merge_datasets(self, X_1, X_2, X_3, y_1, y_2, y_3):
      """
      This function:
      - receives in input three dataset with their labels
      - stratifies them
      - concatenates them (and their labels)
      - shuffles them
      - returns training and test set   
      """
      X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.2, random_state=15, stratify = y_1)
      X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.2, random_state=15, stratify = y_2)
      X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X_3, y_3, test_size=0.2, random_state=15, stratify = y_3)

      X_train = np.concatenate((X_1_train, X_2_train, X_3_train), axis=0)
      X_test = np.concatenate((X_1_test, X_2_test, X_3_test), axis=0)
      y_train = np.concatenate((y_1_train, y_2_train, y_3_train), axis=0)
      y_test = np.concatenate((y_1_test, y_2_test, y_3_test), axis=0)

      X_train, y_train = shuffle(X_train, y_train, random_state=15)
      X_test, y_test = shuffle(X_test, y_test, random_state=15)

      return X_train, X_test, y_train, y_test