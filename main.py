# from _typeshed import StrPath
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from train import NERdataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils_crf import build_vocab, build_dict, cal_max_length, Config
from torch.optim import Adam
import logging
import matplotlib.pyplot as plt
import numpy as np

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_batch(data ,word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad=[]
    tags_pad=[]
    for seq, tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len-len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len-len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_add(args):
    return torch.log(torch.sum(torch.exp(args), axis=0))



class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        begin = time.time()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to('cuda')
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        print('time consuming of crf_partion_function_prepare:%f' % (time.time() - begin))
        begin = time.time()
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = (forward_var + trans_score + emit_score)
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        # print('time consuming of crf_partion_function1:%f' % (time.time() - begin))
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print('time consuming of crf_partion_function2:%f' %(time.time()-begin))
        return alpha

    def _forward_alg_new(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.).to('cuda')
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)  # +1
            aa = gamar_r_l + t_r1_k + self.transitions
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1).cuda()
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha


    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(embeds)
        #lstm_out = lstm_out.view(embeds.shape[1], self.hidden_dim)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        # score = autograd.Variable(torch.Tensor([0])).to('cuda')
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags.view(-1)])

        # if len(tags)<2:
        #     print(tags)
        #     sys.exit(0)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        #feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0]).to('cuda')
        tags = torch.cat([torch.full([tags.shape[0],1],self.tag_to_ix[START_TAG], dtype=torch.long).cuda(),tags],dim=1)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:,-1]]
        return score



    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var.to('cuda') + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_new(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_new(feats)
        gold_score = self._score_sentence(feats, tags)[0]
        return forward_score - gold_score

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq




def val(config, model):

    # ignore the pad label
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    dataloader = DataLoader(testset, batch_size=config.batch_size)
    preds, labels = [], []
    print("Validating...")
    logger.info("Testing...")
    t1 = time.time()
    for index, data in enumerate(dataloader):

        optimizer.zero_grad()
        corpus_, label_, length_ = data
        
        for idx, (corpus, label, length) in enumerate(zip(corpus_, label_, length_)):
            corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
            output = model(corpus)
            score, predict = output
            tmp = []
            for j in label:
                if j.item() < 7:
                    tmp.append(j.item())

            # 不考虑pad的标签。
            preds.extend(predict[:len(tmp)])
            labels.extend(label[:len(tmp)].cpu())

            # predict = torch.argmax(output, dim=-1)
            # loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
    
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)

    logger.info(report)
    print(report)
    print("Time cose:", time.time() - t1)
    model.train()
    return precision, recall, f1


if __name__ == '__main__':

    # ---------- construct logger ------------

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/' 
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.info('Logger constructed!')

    # ----------------------------------------

    config = Config()
    torch.cuda.set_device(config.gpu)

    word_dict = build_vocab(config.data_dir)
    word2id, tag2id = build_dict(word_dict, START_TAG, STOP_TAG)
    max_length = cal_max_length(config.data_dir)
    trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
    dataloader = DataLoader(trainset, batch_size=config.batch_size)
    

    model = BiLSTM_CRF_MODIFY_PARALLEL(len(word2id), tag2id, config.embedding_dim, config.hidden_dim).cuda()

    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = Adam(model.parameters(), config.learning_rate)

    best_f1 = 0.0
    losses = []
    f1_list = []
    for epoch in range(config.epoch): 
        loss_epoch = 0.0
        for idx, data in enumerate(dataloader):

            # corpus, label, length, sent, txt_label = data
            corpus, label, length = data
            corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
    
            model.zero_grad() 

            loss = model.neg_log_likelihood_parallel(corpus, label)
            loss_epoch += loss.item()
            
            loss.backward()
            optimizer.step()

            if idx % 200 == 0 and idx != 0:
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (idx, loss.item()))
                logger.info('[epoch: {:2d}]\t[step: {:4d}]\t[loss: {:.4f}]'.format(epoch, idx, loss.item()))
        prec, rec, f1 = val(config, model)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, config.save_model)
            logger.info("Better model! Best F1: {:.4f}".format(best_f1))
        print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (idx, loss.item()))

        losses.append(loss_epoch)
        f1_list.append(f1)
    
    # draw training curve 
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.xlabel('EPOCH')
    plt.ylabel('total batch loss')
    min_epoch=np.argmin(losses)
    show_min='['+str(min_epoch)+' '+str(losses[min_epoch])+']'
    plt.annotate(show_min,xytext=(min_epoch,losses[min_epoch]),xy=(min_epoch,losses[min_epoch]))
    plt.plot(min_epoch,losses[min_epoch],'gs')
    plt.title('Training loss - epoch')

    plt.subplot(2, 1, 2)
    plt.plot(f1_list)
    plt.xlabel('EPOCH')
    plt.ylabel('total batch loss')
    min_epoch=np.argmax(f1_list)
    show_min='['+str(min_epoch)+' '+str(f1_list[min_epoch])+']'
    plt.annotate(show_min,xytext=(min_epoch,f1_list[min_epoch]),xy=(min_epoch,f1_list[min_epoch]))
    plt.plot(min_epoch,f1_list[min_epoch],'gs')
    plt.title('F1 score - epoch')

    plt.tight_layout()
    plt.savefig('Training_curve.png')