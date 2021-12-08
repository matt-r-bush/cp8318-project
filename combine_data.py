from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
import numpy as np

# open train images
with open('./initialdata/train/questions.json', 'r+') as file:
    trq = json.load(file)

# open train images
with open('./initialdata/test/questions.json', 'r+') as file:
    teq = json.load(file)

# create new object to store combined data
newData ={}

# add all the data from the training set
for t in trq:
    newData[len(newData)] = t

# to keep track of which files have been renamed
finished = []
for t in teq:
    # name of file to rename from
    renameFrom = './initialdata/test/images/'+str(t[2])+'.png'
    t[2] = t[2]+4000
    # add data with new id to newData
    newData[len(newData)] = t
    # name to rename file to
    renameTo = './initialdata/test/images/'+str(t[2])+'.png'
    if not (renameFrom in finished):
        os.rename(renameFrom, renameTo)
        finished.append(renameFrom)

    # print('teq')

with open('./data/questions.JSON', 'w') as outfile:
    json.dump(newData, outfile, indent=4)

outfile.close()