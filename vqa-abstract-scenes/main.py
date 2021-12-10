from tensorflow import keras

## for model building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import argmax
## allows me to choose devices for training
from tensorflow import device

## imports from our files
from data_prep import get_data

## for dealing with images
from skimage.io import imread
from skimage.transform import resize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# from sklearn.metrics import get_precision, get_recall, get_f1, get_accuracy

## for visualization
from keras.utils.vis_utils import plot_model

## helpers
import numpy as np
import json
import metrics

IMG_SHAPE = (200, 350, 3)
BATCH_SIZE = 128

# construct the training image generator for data augmentation
class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, questions, labels, batch_size) :
    self.image_filenames = image_filenames
    self.questions = questions
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_imgs = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_qs = self.questions[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return [np.array([
            resize(imread(file_name), IMG_SHAPE) ## image resize is hardcoded
               for file_name in batch_imgs])/255.0, batch_qs], np.array(batch_labels)



def init_model(img_shape, vocab_size, num_answers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    img_info = Conv2D(8, 3, padding='same')(img_input) ## may need to pad conv2D
    img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Conv2D(16, 3, padding='same')(img_info)
    img_info = MaxPooling2D(padding='same')(img_info)

    img_info = Conv2D(32, 3, padding='same')(img_info)
    img_info = MaxPooling2D(padding='same')(img_info)
    
    ## could add a dropout layer here

    img_info = Flatten()(img_info)
    
    img_info = Dense(32, activation ='swish')(img_info)

    ## question NN ## right now takes word vector inputs but could add an embedding layer to see what happens

    q_input = Input(shape=(vocab_size,), name = 'question_input')
    q_info = Dense(32, activation='swish')(q_input)
    q_info = Dense(32, activation='swish')(q_info)


    ## merge img_info and q_info

    output = Multiply()([img_info, q_info])
    output = Dense(32, activation='swish')(output)
    output = Dense(num_answers, activation='softmax')(output)

    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

train_imgs, test_imgs, train_answers, test_answers, train_qs, test_qs, possible_answers, num_words = get_data()

num_answers = len(possible_answers)

my_training_batch_generator = My_Custom_Generator(train_imgs, train_qs, train_answers, batch_size=BATCH_SIZE)
# my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size=32)

# initialize models for parameters

model = init_model(IMG_SHAPE, num_words, num_answers)


## train model and record history
with device('/cpu:0'):
    history = model.fit_generator(generator = my_training_batch_generator,
        steps_per_epoch = np.ceil(train_answers.shape[0] / BATCH_SIZE), ##32 = batch size
        epochs=10,
        verbose=1
    )

model.save('first_model')

train_imgs = np.load('numpy-arrays/train_imgs.npy')
test_imgs = np.load('numpy-arrays/test_imgs.npy')
train_answers = np.load('numpy-arrays/train_answers.npy')
test_answers = np.load('numpy-arrays/test_answers.npy')
train_qs = np.load('numpy-arrays/train_qs.npy')
test_qs = np.load('numpy-arrays/test_qs.npy')
possible_answers = np.load('numpy-arrays/possible_answers.npy')

model = load_model('first_model')

with device('/cpu:0'):
    predictions = model.predict(x=[test_imgs, test_qs])

print(np.shape(predictions))
print(np.shape(test_answers))

predictions = np.argmax(predictions, axis=1)
test_answers = np.argmax(test_answers, axis=1)

metrics.get_precision(predictions, test_answers) ## using weighted average
metrics.get_recall(predictions, test_answers) ## using weighted average
metrics.get_f1(predictions, test_answers) ## using weighted average
metrics.get_accuracy(predictions, test_answers) ## using weighted average


## could do k-fold cross validation


## evaluate model

# predictions = model.predict([test_imgs, test_qs])
# predictions = np.argmax(predictions, axis=1)
# test_answers = np.argmax(test_answers, axis=1)

# get_precision(predictions, test_answers) ## using weighted average
# get_recall(predictions, test_answers) ## using weighted average
# get_f1(predictions, test_answers) ## using weighted average
# get_accuracy(predictions, test_answers) ## using weighted average

