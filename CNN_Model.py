class CNN_Model:
    def __init__(self, input_shape=(512, 512, 3), num_classes=3):
        self.input_shape=input_shape
        self.num_classes=num_classes

    """
    This function:
    - loads the pre-trained EfficientNetB0 model and exclude the top layers
    - adds a global average pooling layer, a fully connected layer and a
      classification layer
    - freezes the pre-trained layers except the last 5 layers (for fine tuning)
    - compiles the model using Adam as optimizer and sparse categorical crossentropy
      as loss function
    """
    def build_model(self):
        base_model=EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)

        x=base_model.output
        x=layers.GlobalAveragePooling2D()(x)
        x=layers.Dense(128, activation='relu')(x)
        predictions=layers.Dense(self.num_classes, activation='softmax')(x)

        model=models.Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[:-5]:
            layer.trainable = False

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
        self.model=model

    """
    #This function:
    - receives in input the data on which the model has to be trained (images and
      corresponding classes), the number of epochs and the validation split
    - trains the model and returns the history
    """
    def train(self, X_train, y_train, epochs, validation_split):
      self.build_model()

      if len(y_train.shape) == 2:
        y_train = np.argmax(y_train, axis=1)

      self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)
      return self.history

    """
    This function:
     - performs cross-validation, receiving in input the data on which cross validation
       has to be performed and the number of splits
     - the model is trained on 10 epochs with and with a validation split=0.2 for
       every fold
    """
    def fit_cv(self, X, y, n_splits):
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)
        scores=[]
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.build_model()

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.train(X_train, y_train, epochs=10, validation_split=0.2)

            score = self.model.evaluate(X_test, y_test, verbose=0)
            scores.append(score[1])

        return scores 

    """
    This function:
    - predicts the labels of the image data received in input 
    """
    def predict(self, X_test):
      y_pred=self.model.predict(X_test)
      return y_pred   
