import tensorflow as tf
import numpy as np
import pathlib

# Enable NVIDIA GPU for Training
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU found. Ensure TensorFlow is installed with NVIDIA support.")
else:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using NVIDIA GPU: {physical_devices}")

# Load Predefined Train and Validation Directories
data_dir_train = pathlib.Path("afhq/train")
data_dir_val = pathlib.Path("afhq/val")
img_size = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_train, image_size=img_size, batch_size=32)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_val, image_size=img_size, batch_size=32)

# Preprocess Data
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Build ResNet Model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
model.save('animal_species_model.h5')