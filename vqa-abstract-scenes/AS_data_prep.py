## Help with data prep from https://github.com/GT-Vision-Lab/VQA_LSTM_CNN

## for shuffling data
from random import shuffle, seed

## for tokenizing questions
from nltk.tokenize import word_tokenize
import string as s

## helper
import numpy as np
import json




def tokenize(data):
  
    # preprocess all the question
    print('example processed tokens:')
    for i, example in enumerate(data):
        q = example['question']

        ## remove punctuation find a way to remove punctuation


        ## tokenize with nltk

        tokens = word_tokenize(str(q).lower())

        example['q_tokens'] = tokens
        if i < 10:
            print(tokens)
        if i % 1000 == 0:
            print("processing %d/%d (%.2f%% done)   \r" %  (i, len(data), i*100.0/len(data)))

    return data



# def create_vocab():
#     # build vocabulary for question and answers.

#     # count up the number of words
#     counts = {}
#     for img in imgs:
#         for w in img['processed_tokens']:
#             counts[w] = counts.get(w, 0) + 1
#     cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
#     print 'top words and their counts:'
#     print '\n'.join(map(str,cw[:20]))

#     # print some stats
#     total_words = sum(counts.itervalues())
#     print 'total words:', total_words
#     bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
#     vocab = [w for w,n in counts.iteritems() if n > count_thr]
#     bad_count = sum(counts[w] for w in bad_words)
#     print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
#     print 'number of words in vocab would be %d' % (len(vocab), )
#     print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


#     # lets now produce the final annotation
#     # additional special UNK token we will use below to map infrequent words to
#     print 'inserting the special UNK token'
#     vocab.append('UNK')
  
#     for img in imgs:
#         txt = img['processed_tokens']
#         question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
#         img['final_question'] = question

#     return imgs, vocab


# def create_vocab_sequential(data):
#     vocab = {}  
#     index = 1 # start indexing from 1
#     vocab['<pad>'] = 0  # add a padding token
#     for example in data:
#         for token in example['q_tokens']
#             vocab[token] = index


#     inverse_vocab = {index: token for token, index in vocab.items()}

#     return vocab, inverse_vocab

def create_question_vocab(data):
    max_len = 0
    tokens = []
    for example in data:
        current_len = 0
        for j, token in enumerate(example['q_tokens']):
            current_len = j+1
            tokens.append(token)
        if current_len > max_len:
            max_len = current_len

    vocab = {}  
    index = 1 # start indexing from 1 because of padding token
    vocab['placeholder'] = 0  # add a padding token
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1

    inverse_vocab = {index: token for token, index in vocab.items()}

    return vocab, inverse_vocab, max_len

def create_answer_vocab(data): ## two word answers might not matter?
    answers = []
    for example in data:
        answer = example['ans']
        answers.append(answer)
    vocab = {}
    index = 0
    for answer in answers:
        if answer not in vocab:
            vocab[answer] = index
            index += 1

    inverse_vocab = {index: token for token, index in vocab.items()}

    return vocab, inverse_vocab

    ## if we want to have a count threshold
    # count_thr = 50

    # # count up the number of words
    # counts = {}
    # for img in imgs:
    #     for w in img['processed_tokens']:
    #         counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    # print 'top words and their counts:'
    # print '\n'.join(map(str,cw[:20]))

    # # print some stats
    # total_words = sum(counts.itervalues())
    # print 'total words:', total_words
    # bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    # vocab = [w for w,n in counts.iteritems() if n > count_thr]
    # bad_count = sum(counts[w] for w in bad_words)
    # print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    # print 'number of words in vocab would be %d' % (len(vocab), )
    # print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)
    # cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    # print 'top words and their counts:'
    # print '\n'.join(map(str,cw[:20]))

    # # print some stats
    # total_words = sum(counts.itervalues())
    # print 'total words:', total_words
    # bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    # vocab = [w for w,n in counts.iteritems() if n > count_thr]
    # bad_count = sum(counts[w] for w in bad_words)
    # print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    # print 'number of words in vocab would be %d' % (len(vocab), )
    # print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

def prepare_questions(data, vocab, max_len):
    n = len(data)

    print(max_len)

    int_questions = np.zeros((n, max_len), dtype='uint32')
    question_ids = np.zeros(n, dtype='uint32')
    question_counter = 0
    for i,example in enumerate(data):
        question_ids[question_counter] = example['ques_id']
        question_counter += 1
        for j, token in enumerate(example['q_tokens']):
            if token in vocab:
                int_questions[i,j] = vocab[token]    
    
    return int_questions, question_ids

def get_unique_img(imgs): ## change this up
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w,n in count_img.items()]
    imgtoi = {w:i for i,w in enumerate(unique_img)}


    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['img_path'])

    return unique_img, img_pos


def encode_answer(imgs, atoi): ## change this up
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi[img['ans']]

    return ans_arrays

def encode_mc_answer(imgs, atoi): ## change this up
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays




def main():
    raw_train = json.load(open('abstract_train.json', 'r'))
    raw_test = json.load(open('abstract_test.json', 'r'))

    seed(5318008)
    shuffle(raw_train)

    ## tokenize data
    raw_train = tokenize(raw_train)
    raw_test = tokenize(raw_test)

    q_vocab, q_inverse_vocab, max_len = create_question_vocab(raw_train)

    a_vocab, a_inverse_vocab = create_answer_vocab(raw_train)

    ## get numpy arrays of trainable data
    int_qs_train, train_qid = prepare_questions(raw_train, q_vocab, max_len)
    int_qs_test, test_qid = prepare_questions(raw_test, q_vocab, max_len)
    
    ## change this up
    unique_img_train, img_pos_train = get_unique_img(raw_train)
    unique_img_test, img_pos_test = get_unique_img(raw_test)

    ## get encoded answers
    train_answers = encode_answer(raw_train, a_vocab)
    MC_ans_test = encode_mc_answer(raw_test, a_vocab)





if __name__ == '__main__':
    main()