import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# Load the CSV file containing image pairs and labels
csv_path = "train.csv"
data = pd.read_csv(csv_path)

# Get the number of rows corresponding to the first 80% of your dataset
num_samples_80_percent = int(0.8 * len(data))

# Slice the data to use only the first 80% of images
train_data = data.iloc[:num_samples_80_percent]

# Path to the folder with images
image_folder = "train"

# Define the Siamese network architecture
def create_siamese_network(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (5, 5), activation='relu')(input_layer)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    encoded = layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # L2-normalization

    return models.Model(inputs=input_layer, outputs=encoded)

# Create and compile the Siamese model
image_height = 245
image_width = 200
image_channels = 1
input_shape = (image_height, image_width, image_channels)
left_input = layers.Input(shape=input_shape)
right_input = layers.Input(shape=input_shape)

siamese_network = create_siamese_network(input_shape)

encoded_left = siamese_network(left_input)
encoded_right = siamese_network(right_input)

# Define the contrastive loss function
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = K.cast(y_true, 'float32')  # Cast y_true to float32
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0.0)))

# Create the contrastive model
distance = layers.Lambda(lambda embeddings: K.abs(embeddings[0] - embeddings[1]))([encoded_left, encoded_right])
output = layers.Dense(1, activation='sigmoid')(distance)

contrastive_model = models.Model(inputs=[left_input, right_input], outputs=output)
contrastive_model.compile(loss=contrastive_loss, optimizer=Adam(0.0001))

# Define a custom data generator that includes hard negative mining
import random

def siamese_data_generator(data, batch_size, siamese_network, num_hard_negatives=10):
    while True:
        left_images, right_images, labels = [], [], []
        
        for _ in range(batch_size):
            # Randomly select a positive pair
            anchor_index = np.random.randint(0, len(data))
            anchor_left, anchor_right = data.iloc[anchor_index]
            anchor_label = 1.0  # Similar pair

            # Load and preprocess the anchor images
            anchor_left = load_image(anchor_left, is_left=True)
            anchor_right = load_image(anchor_right, is_left=False)

            # Calculate the embeddings for the anchor images
            anchor_left_embedding = siamese_network.predict(np.expand_dims(anchor_left, axis=0))
            anchor_right_embedding = siamese_network.predict(np.expand_dims(anchor_right, axis=0))

            # Randomly sample a subset of dissimilar pairs
            dissimilar_indices = random.sample(range(len(data)), num_hard_negatives)

            # Initialize variables to keep track of the hardest negative and its distance
            hardest_negative = None
            hardest_distance = float('inf')

            for i in dissimilar_indices:
                if i != anchor_index:
                    other_right = load_image(data.iloc[i]['right'], is_left=False)

                    other_right_embedding = siamese_network.predict(np.expand_dims(other_right, axis=0))
                    distance = np.linalg.norm(anchor_left_embedding - anchor_right_embedding) + np.linalg.norm(other_right_embedding - anchor_right_embedding)

                    if distance < hardest_distance:
                        hardest_negative = other_right
                        hardest_distance = distance

            if hardest_negative is not None:
                left_images.append(anchor_left)
                right_images.append(hardest_negative)
                labels.append(anchor_label)

        yield ([np.array(left_images), np.array(right_images)], np.array(labels))


def load_image(image_path, is_left=True):
    folder = "left" if is_left else "right"
    full_path = f"{image_folder}/{folder}/{image_path}.jpg"  # Include subfolder and ".jpg" file extension
    img = tf.keras.preprocessing.image.load_img(full_path, target_size=(image_height, image_width), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img




# Train the Siamese network with efficient data loading and hard negative mining
batch_size = 32
num_epochs = 10
train_generator = siamese_data_generator(train_data, batch_size, siamese_network)
steps_per_epoch = len(train_data) // batch_size

contrastive_model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=0)
siamese_network.save("siamese_model.h5")
