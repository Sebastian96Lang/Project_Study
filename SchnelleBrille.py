import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.src.legacy.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from random import randint

# Load the original MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Function to randomly rotate and shift an image
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

# Function to create a new dataset with randomly rotated and shifted images
def create_rotated_and_shifted_dataset(images, labels):
    rotated_and_shifted_images = np.array([random_rotate_and_shift_image(image) for image in images])
    return rotated_and_shifted_images, labels

def create_shifted_dataset(images, labels):
    shifted_images = np.array([random_shift(image) for image in images])
    return shifted_images, labels

def create_rotated_dataset(images, labels):
    rotated_images = np.array([random_rotate(image) for image in images])
    return rotated_images, labels


# Normalize the pixel values to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode the labels
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Create a list of datasets
datasets = [
    (train_images, test_images, "Normal MNIST"),
    (create_shifted_dataset(train_images, train_labels)[0],create_shifted_dataset(test_images,test_labels)[0], "Shifted Dataset"),
    (create_rotated_dataset(train_images, train_labels)[0],create_rotated_dataset(test_images,test_labels)[0], "Rotated Dataset"),
    (create_rotated_and_shifted_dataset(train_images, train_labels)[0], create_rotated_and_shifted_dataset(test_images, test_labels)[0], "Shifted and Rotated"),
]

for dataset in datasets:
    train_images, test_images, dataset_name = dataset

    # Build the neural network
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images to a 1D array
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model on the rotated and shifted images
    history = model.fit(train_images, train_labels,
                        epochs=10, batch_size=128,
                        validation_data=(test_images, test_labels), verbose=0)

    # Rest of your code for evaluation and visualization remains the same.

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy on {dataset_name}: {test_accuracy * 100:.2f}%')

    # Select 100 random indices from the test set
    num_images_to_show = 100
    random_indices = np.random.randint(0, len(test_images), num_images_to_show)

    # Get the corresponding images and labels
    selected_images = test_images[random_indices]
    selected_labels = test_labels[random_indices]

    # Use the model to predict the labels for the selected images
    predicted_labels = model.predict(selected_images)
    predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert one-hot to class labels

    # Plot the images along with the predicted and true labels
    #plt.figure(figsize=(14, 14))
    #for i in range(num_images_to_show):
    #    plt.subplot(10, 10, i + 1)
    #    plt.imshow(selected_images[i], cmap='gray')
    #    plt.title(f"Predicted: {predicted_labels[i]}\nTrue: {np.argmax(selected_labels[i])}")
    #    plt.axis('off')

    #plt.tight_layout()
    #plt.show()
