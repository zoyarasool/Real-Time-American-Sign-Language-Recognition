import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


DIGITS_DATASET_PATH = os.path.join(
 "D:/3rd Semester/Projects/sign_language_app/ASL/datasets/digits_dataset"
)
IMG_SIZE = 64      
BATCH_SIZE = 32

print("Digit classes:", os.listdir(DIGITS_DATASET_PATH))
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_digits = datagen.flow_from_directory(
    DIGITS_DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_digits = datagen.flow_from_directory(
    DIGITS_DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#Build CNN Model for digits

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

digits_model = Sequential()

digits_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
digits_model.add(MaxPooling2D(2,2))

digits_model.add(Conv2D(64, (3,3), activation='relu'))
digits_model.add(MaxPooling2D(2,2))

digits_model.add(Flatten())
digits_model.add(Dense(128, activation='relu'))
digits_model.add(Dropout(0.5))

digits_model.add(Dense(train_digits.num_classes, activation='softmax'))

digits_model.summary()

digits_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#Training
history_digits = digits_model.fit(
    train_digits,
    validation_data=val_digits,
    epochs=50
)
import json
with open("D:/3rd Semester/Projects/sign_language_app/ASL/lables/digits_labels.json", "w") as f:
    json.dump(train_digits.class_indices, f)
#save model
digits_model.save("D:/3rd Semester/Projects/sign_language_app/ASL/models/asl_digits_model.h5")
#Testing
img_path = "D:/3rd Semester/Projects/sign_language_app/ASL/datasets/digits_dataset/3/hand2_3_left_seg_3_cropped.jpeg"
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = digits_model.predict(img)

digit_labels = list(train_digits.class_indices.keys())
predicted_digit = digit_labels[np.argmax(prediction)]

print("Predicted Digit:", predicted_digit)