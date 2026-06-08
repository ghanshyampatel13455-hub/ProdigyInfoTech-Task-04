import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = "dataset/leapGestRecog"

images = []
labels = []

print("Loading Images...")

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        label = int(gesture.split("_")[0])

        for img_name in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            images.append(img)
            labels.append(label)

print("Images Loaded Successfully!")

print("Unique labels:", sorted(set(labels)))
print("Max label:", max(labels))
print("Total labels:", len(set(labels)))

X = np.array(images)
y = np.array(labels)

# Convert labels 1-10 into 0-9
y = y - 1

print("Unique labels:", np.unique(y))
print("Max label:", np.max(y))

num_classes = int(len(np.unique(y)))
print("Number of classes:", num_classes)

y = to_categorical(y, num_classes=num_classes)

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# CNN Model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(64, 64, 3)
    )
)

model.add(MaxPooling2D(2, 2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "training_history.csv",
    index=False
)

print("Training history saved!")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")

os.makedirs("models", exist_ok=True)

model.save("models/hand_gesture_model.keras")

print("Model Saved Successfully!")

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy_graph.png")
plt.show()

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig("loss_graph.png")
plt.show()