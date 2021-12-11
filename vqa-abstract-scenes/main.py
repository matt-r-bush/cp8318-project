from tensorflow import keras

## for model building
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import argmax
## allows me to choose devices for training
from tensorflow import device

## imports from our files
from other_data_prep import get_data

## for dealing with images
from skimage.io import imread
from skimage.transform import resize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# from sklearn.metrics import get_precision, get_recall, get_f1, get_accuracy

## for visualization
from keras.utils.vis_utils import plot_model

## helpers
import numpy as np
import pickle
import json
import metrics

IMG_SHAPE = (200, 350, 3)
BATCH_SIZE = 128
MODEL_NAME = 'top_10'

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



def init_model(img_shape, vocab_size, num_answers, layers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    if layers >= 1:
      img_info = Conv2D(8, 3, padding='same')(img_input) ## may need to pad conv2D
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 2:
      img_info = Conv2D(16, 3, padding='same', activation ='swish', kernel_initializer='he_uniform')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 3:
      img_info = Conv2D(32, 3, padding='same', activation ='swish', kernel_initializer='he_uniform')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 4:
      img_info = Conv2D(64, 3, padding='same', activation ='swish', kernel_initializer='he_uniform')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)

    if layers >= 5:
      img_info = Conv2D(128, 3, padding='same', activation ='swish', kernel_initializer='he_uniform')(img_info)
      img_info = MaxPooling2D(padding='same')(img_info)
    
    ## could add a dropout layer here

    img_info = Flatten()(img_info)
    
    img_info = Dense(128, activation ='swish', kernel_initializer='he_uniform')(img_info)

    ## question NN ## right now takes word vector inputs but could add an embedding layer to see what happens

    q_input = Input(shape=(vocab_size,), name = 'question_input')
    q_info = Dense(128, activation='swish')(q_input)
    q_info = Dense(128, activation='swish')(q_info)


    ## merge img_info and q_info

    output = Multiply()([img_info, q_info])
    output = Dense(128, activation='swish')(output)
    output = Dense(num_answers, activation='softmax')(output)

    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def run_model(epochs, model_name, layers, check_top_ans, removeYesNo):
  train_imgs, test_imgs, train_answers, test_answers, train_qs, test_qs, possible_answers, num_words = get_data(check_top_ans, removeYesNo)

  num_answers = len(possible_answers)

  my_training_batch_generator = My_Custom_Generator(train_imgs, train_qs, train_answers, batch_size=BATCH_SIZE)
  my_validation_batch_generator = My_Custom_Generator(test_imgs, test_qs, test_answers, batch_size=BATCH_SIZE)

  # initialize models for parameters

  model = init_model(IMG_SHAPE, num_words, num_answers, layers)

  # model = load_model(MODEL_NAME)


  for i in range(epochs):
    ## train model and record history
    # with device('/cpu:0'):
    history = model.fit(x = my_training_batch_generator,
        steps_per_epoch = np.ceil(train_answers.shape[0] / BATCH_SIZE),
        epochs=1,
        verbose=1,
        validation_data = my_validation_batch_generator,
        validation_steps = np.ceil(test_answers.shape[0] / BATCH_SIZE)
    )

    with open('{}_hist'.format(model_name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    model.save(model_name)

# 3 layers, yes/no (top 2)
run_model(epochs=1, model_name='3L_yes_no', layers=3, check_top_ans=2, removeYesNo=False)

# 4 layers, yes/no (top 2)
run_model(epochs=1, model_name='4L_yes_no', layers=4, check_top_ans=2, removeYesNo=False)

# 5 layers, yes/no (top 2)
run_model(epochs=1, model_name='5L_yes_no', layers=5, check_top_ans=2, removeYesNo=False)

# 3 layers, top 10 answers (not including yes/no)
run_model(epochs=1, model_name='3L_top_10', layers=3, check_top_ans=2, removeYesNo=True)

# 4 layers, top 10 answers (not including yes/no)
run_model(epochs=1, model_name='4L_top_10', layers=4, check_top_ans=2, removeYesNo=True)

# 5 layers, top 10 answers (not including yes/no)
run_model(epochs=1, model_name='5L_top_10', layers=5, check_top_ans=2, removeYesNo=True)

# 3 layers, top 100 answers (not including yes/no)
run_model(epochs=1, model_name='3L_top_100', layers=3, check_top_ans=2, removeYesNo=True)

# 4 layers, top 100 answers (not including yes/no)
run_model(epochs=1, model_name='4L_top_100', layers=4, check_top_ans=2, removeYesNo=True)

# 5 layers, top 100 answers (not including yes/no)
run_model(epochs=1, model_name='5L_top_100', layers=5, check_top_ans=2, removeYesNo=True)

# model = load_model(MODEL_NAME)

# with device('/cpu:0'):
# predictions = model.predict(x=my_validation_batch_generator,
#     steps = np.ceil(test_answers.shape[0] / BATCH_SIZE))

# print(np.shape(predictions))
# print(np.shape(test_answers))

# predictions = np.argmax(predictions, axis=1)
# test_answers = np.argmax(test_answers, axis=1)

# metrics.get_precision(predictions, test_answers) ## using weighted average
# metrics.get_recall(predictions, test_answers) ## using weighted average
# metrics.get_f1(predictions, test_answers) ## using weighted average
# metrics.get_accuracy(predictions, test_answers) ## using weighted average


## could do k-fold cross validation


## evaluate model

# predictions = model.predict([test_imgs, test_qs])
# predictions = np.argmax(predictions, axis=1)
# test_answers = np.argmax(test_answers, axis=1)

# get_precision(predictions, test_answers) ## using weighted average
# get_recall(predictions, test_answers) ## using weighted average
# get_f1(predictions, test_answers) ## using weighted average
# get_accuracy(predictions, test_answers) ## using weighted average

