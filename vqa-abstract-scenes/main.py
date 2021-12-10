## for model building
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import argmax

## imports from our files
from data_prep import get_data
# from sklearn.metrics import get_precision, get_recall, get_f1, get_accuracy

## helpers
import numpy as np
import json

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

train_imgs, test_imgs, train_answers, test_answers, train_qs, test_qs, possible_answers, num_words, img_shape = get_data()

num_answers = len(possible_answers)

## initialize models for parameters

model = init_model(img_shape, num_words, num_answers)


## train model and record history

history = model.fit(x=[train_imgs, train_qs],
    y=train_answers,
    epochs=10,
    verbose=1,
    batch_size=8,
)

model.save('first_model')


## could do k-fold cross validation


## evaluate model

# predictions = model.predict([test_imgs, test_qs])
# predictions = np.argmax(predictions, axis=1)
# test_answers = np.argmax(test_answers, axis=1)

# get_precision(predictions, test_answers) ## using weighted average
# get_recall(predictions, test_answers) ## using weighted average
# get_f1(predictions, test_answers) ## using weighted average
# get_accuracy(predictions, test_answers) ## using weighted average

