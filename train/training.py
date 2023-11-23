import os
import shutil
import random
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

dataset_dir = "../TrainingImages"
train_dir = "../Train"
test_dir = "../Test"
validation_dir = "../Validation"

# Create directories for train, test, and validation sets if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

categories = os.listdir(dataset_dir)

for category in categories:
    category_dir = os.path.join(dataset_dir, category)
    images = os.listdir(category_dir)
    random.shuffle(images)

    num_images = len(images)
    train_split = int(0.8 * num_images)
    test_split = int(0.1 * num_images)

    train_images = images[:train_split]
    test_images = images[train_split:train_split + test_split]
    validation_images = images[train_split + test_split:]

    # Create subdirectories for each category in train, test, and validation
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

    for img in train_images:
        src = os.path.join(category_dir, img)
        dst = os.path.join(train_dir, category, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(category_dir, img)
        dst = os.path.join(test_dir, category, img)
        shutil.copy(src, dst)

    for img in validation_images:
        src = os.path.join(category_dir, img)
        dst = os.path.join(validation_dir, category, img)
        shutil.copy(src, dst)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(9, activation='softmax')(x)

model = Model(base_model.input, x)

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1
)

# Save the model
model.save('./bin/citrus_ai_model.h5')


# Test Model
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
