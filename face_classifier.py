import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2 as cv
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

class classifier():
    def __init__(self):
        pass
    
    def train_model(self):
        classifier = Sequential()

        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation = "relu"))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())

        classifier.add(Dense(units = 128, activation = "relu"))
        classifier.add(Dense(units = 128, activation = "relu"))
        classifier.add(Dense(units = 2, activation = "softmax"))

        classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        train_data_generator = ImageDataGenerator(rescale=1./255,
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True,)

        test_data_generator = ImageDataGenerator(rescale=1./255)


        train_set = train_data_generator.flow_from_directory("training/",
                                                            target_size=(64, 64),
                                                            batch_size=10,
                                                            class_mode="categorical")

        test_set = train_data_generator.flow_from_directory("testing/",
                                                            target_size=(64, 64),
                                                            batch_size=10,
                                                            class_mode="categorical")
        name_dic = {number:name for name, number in train_set.class_indices.items()}

        history = classifier.fit_generator(train_set,
                                        nb_epoch=10,
                                        nb_val_samples=1,
                                        steps_per_epoch=10,
                                        verbose=1,
                                        validation_data=test_set)
        return classifier, name_dic
    
    def predict(self, model, name_dic):
        img = Image.open(f"D:/Jordan_and_Butler/testing/two_guys4.jpg")
        image_matrix = cv.imread(f"D:/Jordan_and_Butler/testing/two_guys4.jpg")
        face_cascade = cv.CascadeClassifier(r"C:\Users\User\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(image_matrix, 1.3, 5)
        font = cv.FONT_HERSHEY_COMPLEX
        for x, y, width, height in faces:
            face_box = (x, y, x+width, y+height)
            cropimg = img.crop(face_box).resize((64, 64))
            target_img = image.img_to_array(cropimg)
            target_img = np.expand_dims(target_img, axis=0)
            result = model.predict_classes(target_img)[0]
            cv.putText(image_matrix, name_dic[result], (x, y), font, 1, (13, 58, 209), 4)
        
        plt.figure(figsize=(50,30))
        plt.imshow(cv.cvtColor(image_matrix, cv.COLOR_BGR2RGB))

        return "ok"