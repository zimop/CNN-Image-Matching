import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.applications import ResNet50
from keras import backend as K
from sklearn.model_selection import train_test_split  
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Preload images and labels efficiently
from keras.preprocessing.image import ImageDataGenerator

# Load the CSV file containing image pairs and labels
csv_path = "train.csv"
data = pd.read_csv(csv_path)

# Split the data into training, validation, and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Path to the folder with images
image_folder = "train"

# Define the Siamese network architecture using ResNet-50
from keras.regularizers import l2

def create_siamese_network(input_shape, dropout_rate=0.5, l2_penalty=1e-4):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        if layer.name == 'conv5_block1_out':
            break
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_penalty))(x)
    x = layers.Dropout(dropout_rate)(x)
    encoded = layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # L2-normalization

    return models.Model(inputs=base_model.input, outputs=encoded)


# Create and compile the Siamese model
image_height = 245
image_width = 200
image_channels = 3  # ResNet-50 expects color images (3 channels)
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

datagen = ImageDataGenerator(rescale=1.0 / 255)

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(image_height, image_width))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = datagen.standardize(img)  # Apply rescaling
        images.append(img)
    return np.array(images)

batch_size = 32

# Split the data into training, validation, and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Train data
left_image_paths_train = train_data['left'].apply(lambda x: f"{image_folder}/left/{x}.jpg").values
right_image_paths_train = train_data['right'].apply(lambda x: f"{image_folder}/right/{x}.jpg").values

left_images_train = load_images(left_image_paths_train)
right_images_train = load_images(right_image_paths_train)
labels_train = np.ones(len(left_image_paths_train))  # Similar pairs

# Validation data
left_image_paths_val = val_data['left'].apply(lambda x: f"{image_folder}/left/{x}.jpg").values
right_image_paths_val = val_data['right'].apply(lambda x: f"{image_folder}/right/{x}.jpg").values

left_images_val = load_images(left_image_paths_val)
right_images_val = load_images(right_image_paths_val)
labels_val = np.ones(len(left_image_paths_val))  # Similar pairs

# Generate dissimilar pairs efficiently for training
num_dissimilar_pairs = 4000
dissimilar_indices = np.random.choice(len(train_data), (num_dissimilar_pairs, 2))
dissimilar_left_paths = [f"{image_folder}/left/{train_data.iloc[i]['left']}.jpg" for i in dissimilar_indices[:, 0]]
dissimilar_right_paths = [f"{image_folder}/right/{train_data.iloc[i]['right']}.jpg" for i in dissimilar_indices[:, 1]]

dissimilar_left_images = load_images(dissimilar_left_paths)
dissimilar_right_images = load_images(dissimilar_right_paths)
dissimilar_labels = np.zeros(len(dissimilar_left_paths))  # Dissimilar pairs

left_images_train = np.concatenate([left_images_train, dissimilar_left_images], axis=0)
right_images_train = np.concatenate([right_images_train, dissimilar_right_images], axis=0)
labels_train = np.concatenate([labels_train, dissimilar_labels], axis=0)

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Train the Siamese network with efficient data loading and pairs, using validation data and early stopping
history = contrastive_model.fit(
    [left_images_train, right_images_train],
    labels_train,
    batch_size=batch_size,
    epochs=10,
    validation_data=([left_images_val, right_images_val], labels_val),
    callbacks=[early_stopping]
)
# Optionally, you can save the model
siamese_network.save("siamese_model.h5")

# Testing data
left_image_paths_test = test_data['left'].apply(lambda x: f"{image_folder}/left/{x}.jpg").values
right_image_paths_test = test_data['right'].apply(lambda x: f"{image_folder}/right/{x}.jpg").values

left_images_test = load_images(left_image_paths_test)
right_images_test = load_images(right_image_paths_test)
labels_test = np.ones(len(left_image_paths_test))  # Similar pairs

# Generate dissimilar pairs efficiently for testing
num_dissimilar_pairs_test = 1000  # You can adjust the number of dissimilar pairs for testing
dissimilar_indices_test = np.random.choice(len(test_data), (num_dissimilar_pairs_test, 2))
dissimilar_left_paths_test = [f"{image_folder}/left/{test_data.iloc[i]['left']}.jpg" for i in dissimilar_indices_test[:, 0]]
dissimilar_right_paths_test = [f"{image_folder}/right/{test_data.iloc[i]['right']}.jpg" for i in dissimilar_indices_test[:, 1]]

dissimilar_left_images_test = load_images(dissimilar_left_paths_test)
dissimilar_right_images_test = load_images(dissimilar_right_paths_test)
dissimilar_labels_test = np.zeros(len(dissimilar_left_paths_test))  # Dissimilar pairs

left_images_test = np.concatenate([left_images_test, dissimilar_left_images_test], axis=0)
right_images_test = np.concatenate([right_images_test, dissimilar_right_images_test], axis=0)
labels_test = np.concatenate([labels_test, dissimilar_labels_test], axis=0)

# Evaluate the Siamese network on the testing data
testing_loss = contrastive_model.evaluate([left_images_test, right_images_test], labels_test)
print(f"Testing Loss: {testing_loss}")


