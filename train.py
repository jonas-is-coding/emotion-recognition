from datasets import load_dataset
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Datensatz laden
ds = load_dataset("Jeneral/fer-2013")

# 2. Bilder und Labels extrahieren und dekodieren
def preprocess_example(example):
    # Bilddekodierung
    image = Image.open(io.BytesIO(example['img_bytes']))  # Bild aus Bytes dekodieren
    image = image.convert('L')  # In Graustufen umwandeln (da es nur ein Kanal ist)
    image = image.resize((48, 48))  # Größe auf 48x48 Pixel anpassen
    image = np.array(image)  # In NumPy-Array umwandeln
    image = image.reshape(48, 48, 1)  # Umformen für das CNN
    label = example['labels']
    return image, label

# Train- und Test-Split erstellen
train_images = []
train_labels = []
for example in ds['train']:
    image, label = preprocess_example(example)
    train_images.append(image)
    train_labels.append(label)

# Umwandeln in NumPy-Arrays und Normalisieren
train_images = np.array(train_images) / 255.0  # Normalisieren auf Werte zwischen 0 und 1
train_labels = to_categorical(np.array(train_labels), num_classes=7)

# Testdaten vorbereiten
test_images = []
test_labels = []
for example in ds['test']:
    image, label = preprocess_example(example)
    test_images.append(image)
    test_labels.append(label)

test_images = np.array(test_images) / 255.0
test_labels = to_categorical(np.array(test_labels), num_classes=7)

# 3. Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 Klassen
])

# Modell kompilieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Modell trainieren
model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_data=(test_images, test_labels))

model.save('emotion_detection_model.h5')  # Modell speichern
