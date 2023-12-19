import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from random import randint

# Load the original MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Function to randomly rotate an image
def random_rotate_image(image):
    angle = np.random.randint(-170, 170)  # Random angle between -30 and 30 degrees
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for the generator
    return np.squeeze(ImageDataGenerator().apply_transform(image, {'theta': angle}), axis=-1)

# Function to create a new dataset with randomly rotated images
def create_rotated_dataset(images, labels):
    rotated_images = np.array([random_rotate_image(image) for image in images])
    return rotated_images, labels

# Create the new rotated datasets
rotated_train_images, train_labels = create_rotated_dataset(train_images, train_labels)
rotated_test_images, test_labels = create_rotated_dataset(test_images, test_labels)

# Plot a few examples of the rotated images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(rotated_train_images[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.show()


# Normalize the pixel values to the range [0, 1]
rotated_train_images = rotated_train_images.astype('float32') / 255.0
rotated_test_images = rotated_test_images.astype('float32') / 255.0

# One-hot encode the labels
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Build the neural network
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images to a 1D array
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the rotated images
history = model.fit(rotated_train_images, train_labels,
                    epochs=10, batch_size=128,
                    validation_data=(rotated_test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(rotated_test_images, test_labels, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Select 100 random indices from the test set
num_images_to_show = 100
random_indices = np.random.randint(0, len(rotated_test_images), num_images_to_show)

# Get the corresponding images and labels
selected_images = rotated_test_images[random_indices]
selected_labels = test_labels[random_indices]

# Use the model to predict the labels for the selected images
predicted_labels = model.predict(selected_images)
predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert one-hot to class labels

# Plot the images along with the predicted and true labels
plt.figure(figsize=(14, 14))
for i in range(num_images_to_show):
    plt.subplot(10, 10, i + 1)
    plt.imshow(selected_images[i], cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]}\nTrue: {np.argmax(selected_labels[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()
