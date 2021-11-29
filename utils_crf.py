import os
import numpy as np

# config for training
class Config():

    def __init__(self):

        self.learning_rate = 1e-4
        self.dropout = 0.9
        self.epoch = 20
        self.batch_size = 64
        self.data_dir = 'data/'
        self.embedding_dim = 300
        self.hidden_dim = 512
        self.save_model = 'NERmodel.pth'
        self.gpu = 1
        



def build_vocab(data_dir):
    """
    :param data_dir: the dir of train_corpus.txt
    :return: the word dict for training
    """

    if(os.path.isfile('word_dict.npy')):
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        return word_dict
    else:
        word_dict = {}
        train_corpus = data_dir + 'train' +'_corpus.txt'
        lines = open(train_corpus).readlines()
        for line in lines:
            word_list = line.split()
            for word in word_list:
                if(word not in word_dict):
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
        np.save('word_dict.npy', word_dict)
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        return word_dict

def build_dict(word_dict, start_tag, stop_tag):
    
    """
    :param word_dict:
    :return: word2id and tag2id
    """
    
    # 7 is the label of pad
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7, start_tag: 8, stop_tag: 9}
    word2id = {}
    for key in word_dict:
        word2id[key] = len(word2id)
    word2id['unk'] = len(word2id)
    word2id['pad'] = len(word2id)
    return word2id, tag2id

def cal_max_length(data_dir):
    """
    :return: the max length of sentences
    """
    file = data_dir + 'train' + '_corpus.txt'
    lines = open(file).readlines()
    max_len = 0
    for line in lines:
        if(len(line.split()) > max_len):
            max_len = len(line.split())

    return max_len