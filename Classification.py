import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json

# training and testing dataset path
#to download dataset and obtain the paths 
'''
use the code from kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sriramr/f/Users/talhabinomar/ruits-fresh-and-rotten-for-classification")

print("Path to dataset files:", path)
'''
train_dir = '/Users/talhabinomar/.cache/kagglehub/datasets/sriramr/fruits-fresh-and-rotten-for-classification/versions/1/dataset/train'
test_dir = '/Users/talhabinomar/.cache/kagglehub/datasets/sriramr/fruits-fresh-and-rotten-for-classification/versions/1/dataset/test'

#batch size and image resizing size of 150*150 to detect features
img_height, img_width = 150, 150
batch_size = 32
#images augmentation
#of training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

#using data generators to process data in chunks to handle memory overflow issues
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# converting generators to tf.data.Dataset
def generator_to_dataset(generator, steps):
    return tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32)
        )
    ).repeat()


steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

train_dataset = generator_to_dataset(train_generator, steps_per_epoch)
val_dataset = generator_to_dataset(validation_generator, validation_steps)

print("Training samples are : ",train_generator.samples)
print("Validation samples are : ",validation_generator.samples)
print("Test samples are : ",test_generator.samples)
print("Class mapping are : ", train_generator.class_indices)

model = Sequential([
    # Input Layer
    Input(shape=(img_height, img_width, 3)),

    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'), 
    Dropout(0.5), 

    # Output Layer
    Dense(6, activation='softmax') # 6 output classes
])

model.compile(
    optimizer=Adam(learning_rate=0.0001), #Learning rate is set to 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy') #using early stopping incase the model starts to overfit
]

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
    
)
#accuracy and Loss visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

#calculing test accuracy by testing on unseen data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

save_path = "/Users/talhabinomar/Downloads/Power BI/MODEL"
model.save(f"{save_path}/fruit_classification_model_3.keras")
print(f"Model saved to: {save_path}/fruit_classification_model_3.keras")

with open(f"{save_path}/class_indices3.json", 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f"Class indices saved to: {save_path}/class_indices2.json")
