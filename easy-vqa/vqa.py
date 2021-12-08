## for model building
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply

## imports from our files
from data_prep import get_data
from metrics import get_precision, get_recall, get_f1, get_accuracy

def init_model(img_shape, vocab_size, num_answers):

    ## convolutional NN for images
    img_input = Input(shape = img_shape, name ='image_input')
    
    img_info = Conv2D(32, 5, activation ='relu')(img_input) ## may need to pad conv2D
    img_info = MaxPooling2D(5)(img_info)
    
    img_info = Conv2D(64, 5, activation ='relu')(img_info)
    img_info = MaxPooling2D(5)(img_info)
    
    img_info = Conv2D(128, 5, activation ='relu')(img_info)
    img_info = MaxPooling2D(5)(img_info)
    
    img_info = Conv2D(64, 5, activation ='relu')(img_info)
    img_info = MaxPooling2D(5)(img_info)
    
    img_info = Conv2D(32, 5, activation ='relu')(img_info)
    img_info = MaxPooling2D(5)(img_info)
    
    ## could add a dropout layer here

    img_info = Flatten()(img_info)
    
    img_info = Dense(32, activation ='softmax')(img_info)

    ## question NN ## right now takes word vector inputs but could add an embedding layer to see what happens

    q_input = Input(shape=(vocab_size,))
    q_info = Dense(32, activation='relu')(q_input)
    q_info = Dense(32, activation='relu')(q_info)


    ## merge img_info and q_info

    output = Multiply([img_info, q_info])
    output = Dense(32, activation='relu')(output)
    output = Dense(num_answers, activation='softmax')(output)

    model = Model(inputs=[img_input, q_input], outputs=output)
    model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


model = init_model(img_shape, vocab_size, num_answers)


## train model and record history

history = model.fit(x=[train_imgs, train_qs],
    y=train_answers,
    epochs=10,
    verbose=1,
    batch_size=16,
)

model.save()


## could do k-fold cross validation


## evaluate model

predictions = model.predict([test_imgs, test_qs])

