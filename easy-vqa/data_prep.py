from numpy.lib.type_check import _imag_dispatcher
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from random import seed, shuffle
import json
import os
import numpy as np
from pathlib import Path
import math

QUESTIONS_PATH = './data/questions.JSON'
ANSWERS_PATH = './data/answers.txt'
IMAGES_PATH = './data/images/'

def get_data():
    # open questions
    with open(QUESTIONS_PATH, 'r') as file:
        qs = json.load(file)
    file.close()
    # go through the questions file and get the questions, answers, and ids
    questions = []
    ans = []
    ids = []

    for q in qs:
        questions.append(qs[q][0])
        ans.append(qs[q][1])
        ids.append(qs[q][2])

    # open answers file and get all possible answers
    all_ans = []
    total_ans = 0
    with open(ANSWERS_PATH, 'r') as file:
        for a in file:
            all_ans.append(a.strip())
            total_ans+=1
    file.close()

    # map image ids to image paths
    image_paths = {}
    for image in os.listdir(IMAGES_PATH):
        img_id = int(image.split('.')[0])
        image_paths[img_id] = IMAGES_PATH+str(image)

    # map image ids to processed images
    images = {}
    for i in image_paths:
        # load image using keras
        ld_img = load_img(image_paths[i])
        # image to array using keras
        img_arr = img_to_array(ld_img)
        # shift pixel values
        shift_img = img_arr/255-.5
        # save img to images object
        images[i] = shift_img
    # get images shape 
    # image_shape = images[0].shape

    # get all unique words in questions
    words = set('')
    for q in questions:
        # remove question mark
        q = q[:-1]
        words.update(q.split())
    words_len = len(words)

    # tokenize words
    tokens = {}
    counter = 0
    for word in words:
        tokens[word] = counter
        counter += 1

    # bag of words for questions
    # create numpy array of zeroes to hold bags of words for questions
    # size number of questions x number of total words
    questions_bow = np.zeros((len(questions),len(tokens)))
    count = 0
    for q in questions:
        # bag of words for this question
        bow = []
        # remove questions mark
        q = q[:-1]
        # turn into list
        q = q.split()
        for t in tokens:
            # check if the current word from tokens is in the question
            if t in q:
                # if it is, set np array of index count x id of t in token to 1
                questions_bow[count][tokens[t]] = 1.
        count += 1

    # create model inputs
    x = np.array([images[id] for id in ids])

    # create model outputs
    answers_idx = [all_ans.index(a) for a in ans]
    y = to_categorical(answers_idx)

    # zip and shuffle

    xy = list(zip(x, y)) ## zip to keep data and labels aligned during shuffle

    seed(13) ## seed for reproducibility

    shuffle(xy)

    x, y = zip(*xy) ## unzip shuffled data

    # set ratio for train
    ratio = 0.8
    # split x into train and test
    ratio = math.floor(len(x)*ratio)
    x_train = x[:ratio]
    x_test = x[ratio:]
    # split x into train and test
    y_train = y[:ratio]
    y_test = y[ratio:]

    return x_train, x_test, y_train, y_test