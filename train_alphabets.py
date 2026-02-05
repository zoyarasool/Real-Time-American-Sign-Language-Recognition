import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

DATASET_PATH = os.path.join(
   "D:/3rd Semester/Projects/sign_language_app/ASL/datasets/alphabets_dataset"
)
print("Dataset path:", DATASET_PATH)
print("Dataset exists:", os.path.exists(DATASET_PATH))
print("Classes:", os.listdir(DATASET_PATH))
print("Total classes:", len(os.listdir(DATASET_PATH)))

IMG_SIZE = 64      
BATCH_SIZE = 32
EPOCHS = 50      

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
#Building CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_data.num_classes, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Train the model for alphabets
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
import json

with open("D:/3rd Semester/Projects/sign_language_app/ASL/lables/alphabet_labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

model.save("D:/3rd Semester/Projects/sign_language_app/ASL/models/asl_alphabets_model.h5")

#testing for alphabets
from tensorflow.keras.preprocessing import image

img_path = "D:/3rd Semester/Projects/sign_language_app/ASL/datasets/alphabets_dataset/b/hand2_b_left_seg_3_cropped.jpeg"

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

class_labels = list(train_data.class_indices.keys())
predicted_class = class_labels[np.argmax(prediction)]

print("Predicted Sign:", predicted_class)