import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ✅ Set dataset path (Use Kaggle’s dataset feature)
train_dir = "/kaggle/input/pokemon-image-dataset"

# ✅ Define model parameters
img_width, img_height = 150, 150
batch_size = 32
epochs = 15
num_classes = 5  # Adjust based on the number of Pokémon classes

# ✅ Load dataset efficiently using TensorFlow’s built-in function
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

# ✅ Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ✅ Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ✅ Save best model automatically
checkpoint_cb = ModelCheckpoint("pokemon_classifier_best.h5", save_best_only=True)

# ✅ Train model with checkpoint
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[checkpoint_cb]
)

# ✅ Save class names
class_names = train_ds.class_names
with open('pokemon_classes.json', 'w') as f:
    json.dump(class_names, f)

print(f"Model saved with classes: {class_names}")
