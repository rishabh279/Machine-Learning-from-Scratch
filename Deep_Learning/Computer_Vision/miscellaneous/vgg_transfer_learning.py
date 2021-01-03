from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import yaml


with open('../utils/default.yaml') as file:
    params = yaml.full_load(file)

train_path = '../data/fruits-360/Training'
valid_path = '../data/fruits-360/Test'


image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

folders = glob(train_path + '/*')

vgg = VGG16(input_shape=params['IMAGE_SIZE'] + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
predictions = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=predictions)

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_gen = gen.flow_from_directory(valid_path, target_size=params['IMAGE_SIZE'])

labels = [None] * len(test_gen.class_indices)

for k, v in test_gen.class_indices.items():
    labels[v] = k

for x, y in test_gen:
    print("min", x[0].min(), "max:", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

# train_generator = gen.flow_from_directory(
#     train_path,
#     target_size=params['IMAGE_SIZE'],
#     shuffle=True,
#     batch_size=params['batch_size']
# )
#
# valid_generator = gen.flow_from_directory(
#     valid_path,
#     target_size=params['IMAGE_SIZE'],
#     shuffle=True,
#     batch_size=params['batch_size']
# )
#
# vgg_model = model.fit_generator(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=params['epochs'],
#     steps_per_epoch=len(image_files) // params['batch_size'],
#     validation_steps=len(valid_image_files) // params['batch_size']
# )

