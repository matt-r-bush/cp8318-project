# Import Keras 
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential


## help from: https://towardsdatascience.com/deep-learning-and-visual-question-answering-c8c8093941bc

# Define CNN for Image Input
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# Define RNN for language input
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Combine CNN and RNN to create the final model
merged = keras.layers.concatenate([encoded_question, encoded_image])
output = Dense(1000, activation='softmax')(merged)
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# split image_feature (from CNN) to a set of features
feature_set = []
for idx in range(NUM_FEATURES):
    feature_unit = Lambda(lambda x : x[:,:,:,idx], output_shape=input_shape[1:3])(image_feature)
    feature_set.append(feature_unit)
    
# combine feature x feature into a set of relations
relation_set = []
for f_i in feature_set:
    for f_j in feature_set:
        if (f_i != f_j):
            relation_unit = keras.layers.concatenate([f_i, f_j])
            relation_set.append(relation_unit)