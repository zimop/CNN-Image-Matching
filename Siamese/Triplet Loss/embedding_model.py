import os
import random
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Concatenate, Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers, models
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization

tf.config.run_functions_eagerly(True)

# Load your CSV file containing pairs of similar images
csv_path = "train.csv"
data = pd.read_csv(csv_path)

# Path to your 'left' and 'right' image folders
left_folder = 'train/left/'
right_folder = 'train/right/'
input_shape = (240, 200, 3)

# Create a preprocessing function
def preprocess_image(image):
    image = image.resize((200, 240), Image.LANCZOS)
    image = np.array(image) / 255.0  # Normalize to the range [0, 1]
    return image

# Function to generate triplets
def generate_triplets(data, left_folder, right_folder):
    triplets = []

    for index, row in data.iterrows():
        left_image = row['left'] + '.jpg'
        right_image = row['right'] + '.jpg'

        # Randomly select a dissimilar image from the 'right' folder
        dissimilar_image = random.choice(os.listdir(right_folder))

        triplets.append((left_image, right_image, dissimilar_image))
        # Display the first triplet
        if len(triplets) == 1:
            print("First generated triplet:", triplets)

    return triplets

# Generate triplets and split into training and validation sets
triplets = generate_triplets(data, left_folder, right_folder)

triplets_train, triplets_val = train_test_split(triplets, test_size=0.2, random_state=42)



# Define a data generator
def data_generator(triplets, batch_size):
    while True:
        anchor_images = []
        positive_images = []
        negative_images = []

        for _ in range(batch_size):
            triplet = random.choice(triplets)
            anchor_image = preprocess_image(Image.open(os.path.join(left_folder, triplet[0])))
            positive_image = preprocess_image(Image.open(os.path.join(right_folder, triplet[1])))
            negative_image = preprocess_image(Image.open(os.path.join(right_folder, triplet[2])))

            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_images.append(negative_image)

        yield [np.array(anchor_images), np.array(positive_images), np.array(negative_images)], np.zeros(batch_size)

# Load the ResNet-50 model with pre-trained weights, excluding the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
# Freeze layers up to conv5_block1_out
for layer in base_model.layers:
    if layer.name == 'conv5_block1_out':
        break
    layer.trainable = False

# Create the Siamese model with shared dense layers
shared_dense = Dense(256, activation='relu')

input_anchor = Input(shape=input_shape, name='anchor_input')
input_similar = Input(shape=input_shape, name='similar_input')
input_dissimilar = Input(shape=input_shape, name='dissimilar_input')
batch_norm = BatchNormalization()  # Add BatchNormalization layer

# Apply shared dense layers
embedding_anchor = shared_dense(base_model(input_anchor))
embedding_similar = shared_dense(base_model(input_similar))

embedding_dissimilar = shared_dense(base_model(input_dissimilar))



siamese_model = Model(
    inputs=[input_anchor, input_similar, input_dissimilar],
    outputs=[embedding_anchor, embedding_similar, embedding_dissimilar]
)
# Create the Siamese model
siamese_model = Model(inputs=[input_anchor, input_similar, input_dissimilar],
                      outputs=[embedding_anchor, embedding_similar, embedding_dissimilar])
# Create the embedding model
embedding_model = Model(inputs=input_anchor, outputs=embedding_anchor)

def triplet_loss(y_true, y_pred):
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]

    # Calculate squared Euclidean distances
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Calculate the triplet loss
    loss = tf.maximum(distance_positive - distance_negative + 0.5, 0.0)

    # Compute the mean over all triplets
    triplet_loss = tf.reduce_mean(loss)

    return triplet_loss


siamese_model.compile(loss=triplet_loss, optimizer='adam')
# Train the model using the data generator
batch_size = 32
epochs = 10
steps_per_epoch = len(triplets_train) // batch_size

patience = 2
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

siamese_model.fit(
    data_generator(triplets_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=data_generator(triplets_val, batch_size),
    validation_steps=len(triplets_val) // batch_size,
    callbacks=[early_stopping],
)

siamese_model.save_weights('triplet_loss_weights.h5')
embedding_model.save('embedding_model.h5')