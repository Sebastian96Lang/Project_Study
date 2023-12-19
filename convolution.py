import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from random import randint

# Load the original MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Function to randomly rotate and shift an image
def random_rotate_and_shift_image(image):
    angle = np.random.randint(-30, 30)  # Random angle between -30 and 30 degrees
    shift = np.random.randint(-5, 5, size=2)  # Random shift in both x and y directions
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for the generator
    return np.squeeze(ImageDataGenerator().apply_transform(image, {'theta': angle, 'tx': shift[0], 'ty': shift[1]}), axis=-1)

# Function to create a new dataset with randomly rotated and shifted images
def create_rotated_and_shifted_dataset(images, labels):
    rotated_and_shifted_images = np.array([random_rotate_and_shift_image(image) for image in images])
    return rotated_and_shifted_images, labels

# Create the new rotated and shifted datasets
rotated_shifted_train_images, train_labels = create_rotated_and_shifted_dataset(train_images, train_labels)
rotated_shifted_test_images, test_labels = create_rotated_and_shifted_dataset(test_images, test_labels)

# Normalize the pixel values to the range [0, 1]
rotated_shifted_train_images = rotated_shifted_train_images.astype('float32') / 255.0
rotated_shifted_test_images = rotated_shifted_test_images.astype('float32') / 255.0

# One-hot encode the labels
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Build the neural network with Conv2D layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the rotated and shifted images
history = model.fit(rotated_shifted_train_images, train_labels,
                    epochs=10, batch_size=128,
                    validation_data=(rotated_shifted_test_images, test_labels))

# Rest of your code for evaluation and visualization remains the same.

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(rotated_shifted_test_images, test_labels, verbose=0)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Select 100 random indices from the test set
num_images_to_show = 100
random_indices = np.random.randint(0, len(rotated_shifted_test_images), num_images_to_show)

# Get the corresponding images and labels
selected_images = rotated_shifted_test_images[random_indices]
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