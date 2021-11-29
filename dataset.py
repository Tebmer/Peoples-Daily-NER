import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import build_vocab, build_dict, cal_max_length, Config
from model import NERLSTM
from torch.optim import Adam, SGD
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.cuda.set_device(1)

class NERdataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()        
        label = open(label_file).readlines()
        self.sents = corpus
        self.txt_labels = label
        self.corpus = []
        self.label = []
        self.length = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            self.length.append(len(corpus_.split()))
            if(len(self.corpus[-1]) > max_length):
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
            else:
                while(len(self.corpus[-1]) < max_length):
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag2id['PAD'])

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.length[item]#, self.sents[item], self.txt_labels[item]

    def __len__(self):
        return len(self.label)
