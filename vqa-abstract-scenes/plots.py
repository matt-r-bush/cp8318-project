import pickle
import os
from matplotlib import pyplot as plt

DIR = "./history/"

for filename in os.listdir(DIR):
    FILENAME = filename
    with open(DIR+FILENAME, 'rb') as handle:
        history = pickle.load(handle)

    print(history)
    #  "Accuracy"
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./plots/'+FILENAME+'.pdf')
    plt.clf()

# # plot model
# import numpy as np
# from tensorflow.python.layers import base
# import tensorflow as tf

# model_path = "./models/cum/"
# model = tf.keras.models.load_model(model_path)
# print('model', model)

# tf.keras.utils.plot_model(model, to_file='cumplot.png', show_shapes=True)


# import tensorflow as tf
# from matplotlib import pyplot as plt
# import pandas as pd

# model_path = "./models/cum/"
# model = tf.keras.models.load_model(model_path)

# print(model.evaluate())

# # history = model

# # print(history.keys())

# # #  "Accuracy"
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'validation'], loc='upper left')
# # plt.show()
# # # "Loss"

# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'val'], loc='upper left')
# # plt.show()