from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

# Image size
IMAGE_SIZE = [224, 224]

# Path to training and test data
train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

# Load VGG16 model pre-trained on ImageNet, without the top classification layers
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all the layers of the pre-trained model
for layer in vgg.layers:
    layer.trainable = False

# Get number of output classes (number of people, i.e., Amit and Aryan)
folders = glob('Datasets/train' + '/*')

# Custom layers
x = Flatten()(vgg.output)
output = Dense(len(folders), activation='softmax')(x)

# Create the model
model = Model(inputs=vgg.input, outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data augmentation for training and test datasets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and test data
train_set = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')

# Train the model
r = model.fit(
    train_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(train_set),
    validation_steps=len(test_set)
)

# Save the trained model
model.save('facefeatures_new_model.h5')

# Plot the training loss and accuracy
import matplotlib.pyplot as plt

# Loss plot
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Accuracy plot
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
