import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.src.legacy.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from random import randint
import time

# Importieren des MNIST Datensatzes via Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Funktionen um den MNIST Datensatzt zu versetzen und rotieren (+Beides auf einmal)
def random_rotate_and_shift_image(image):
    angle = np.random.randint(-110, 110)  # Random angle between -30 and 30 degrees
    shift = np.random.randint(-8, 8, size=2)  # Random shift in both x and y directions
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for the generator
    return np.squeeze(ImageDataGenerator().apply_transform(image, {'theta': angle, 'tx': shift[0], 'ty': shift[1]}), axis=-1)

def random_shift(image):
    shift = np.random.randint(-8, 8, size=2)  # Random shift in both x and y directions
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for the generator
    return np.squeeze(ImageDataGenerator().apply_transform(image, {'theta': 0, 'tx': shift[0], 'ty': shift[1]}), axis=-1)

def random_rotate(image):
    angle = np.random.randint(-110, 110)  # Random angle between -30 and 30 degrees
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for the generator
    return np.squeeze(ImageDataGenerator().apply_transform(image, {'theta': angle, 'tx': 0, 'ty': 0}), axis=-1)

def create_rotated_and_shifted_dataset(images, labels):
    rotated_and_shifted_images = np.array([random_rotate_and_shift_image(image) for image in images])
    return rotated_and_shifted_images, labels

def create_shifted_dataset(images, labels):
    shifted_images = np.array([random_shift(image) for image in images])
    return shifted_images, labels

def create_rotated_dataset(images, labels):
    rotated_images = np.array([random_rotate(image) for image in images])
    return rotated_images, labels


# Normalisieren der Pixel (Graustufen von 0 bis 255) [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Aufteilen der Datensätze in Trainings- und Testdaten
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Liste mit allen vier Datensätzen (Normal, Versetzt, Rotiert, Beides)
datasets = [
    (train_images, test_images, "Normal MNIST"),
    (create_shifted_dataset(train_images, train_labels)[0],create_shifted_dataset(test_images,test_labels)[0], "Shifted Dataset"),
    (create_rotated_dataset(train_images, train_labels)[0],create_rotated_dataset(test_images,test_labels)[0], "Rotated Dataset"),
    (create_rotated_and_shifted_dataset(train_images, train_labels)[0], create_rotated_and_shifted_dataset(test_images, test_labels)[0], "Shifted and Rotated"),
]
# Timer
start_time = time.time()

# Variable für Aktivierungsfunktionen (Nur für Dense)
activation_func = 'linear'

for dataset in datasets:
    train_images, test_images, dataset_name = dataset

    # Initialisiere das Neurale Netzwerk
    model = Sequential()
    # Erste und zweite Konv./Pooling Layer (+ Zeropadding)
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Dropout- und erste Dense Layer
    model.add(Dropout(0.2))
    model.add(Dense(128, activation=activation_func))

    # Dritte Layer mit Konv./Pooling (+ Zeropadding)
    model.add(Dense(7*7*32,activation=activation_func))
    model.add(Reshape((7,7,32)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Zweite Dropout und Dense Layer + finale Layer mit softmax
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=activation_func))
    model.add(Dense(num_classes, activation='softmax'))


    # Modell kompilieren
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Trainieren des Modells mit dem aktuellen Datensatz (normal,shifted,rotated and both)
    history = model.fit(train_images, train_labels,
                        epochs=10, batch_size=128,
                        validation_data=(test_images, test_labels), verbose=0)

    # Evaluieren des Modells (Gleicher Datensatz wie das Training)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Genauigkeit des Netzwerkes mit dem "{dataset_name}" Datensatz: {test_accuracy * 100:.2f}%')

    # 100 zufällige Ergebisse wählen (zum Plotten)
    num_images_to_show = 100
    random_indices = np.random.randint(0, len(test_images), num_images_to_show)

    # Bilder und dazugehörige Label laden
    selected_images = test_images[random_indices]
    selected_labels = test_labels[random_indices]

    # Vorhersagen mit Hilfe des Modells treffen
    predicted_labels = model.predict(selected_images)
    predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert one-hot to class labels

    # Plotten der 100 zufällig ausgewählten Testbeispiele
    #plt.figure(figsize=(14, 14))
    #for i in range(num_images_to_show):
    #    plt.subplot(10, 10, i + 1)
    #    plt.imshow(selected_images[i], cmap='gray')
    #    plt.title(f"Predicted: {predicted_labels[i]}\nTrue: {np.argmax(selected_labels[i])}")
    #    plt.axis('off')

    #plt.tight_layout()
    #plt.show()

# Timer
end_time = time.time()
time_diff = end_time - start_time
print(f"Die Simulation hat {time_diff/60} Minuten gedauert")