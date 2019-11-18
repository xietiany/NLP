import torch
import csv
import torchvision
import torch.nn as nn
import torchvision
from random import shuffle
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import string
import gensim

class BoWClassifier(nn.Module):
    def __init__(self, num_labels, embedding_dim, vocab_size):
        super(BoWClassifier, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.linear = nn.Sequential(
                nn.Linear(embedding_dim, num_labels),
        )
        
    def forward(self, bow_vec):
        embeds = self.embeddings(bow_vec).view(1, 1, -1)
        # add the average pooling
        average_pool = nn.AvgPool1d(len(embeds), len(bow_vec))
        final_input = average_pool(embeds).view((1, -1))
        return torch.sigmoid(self.linear(final_input))

def main():
    #here is the Problem one model/ # word_to_ix is bag_of_word_dic
    label, sentences, text = pre_process('data/clean_data_1.csv')
        
    # print(sentences)
    bag_of_word_dic, size = create_dic(sentences)
    # print(bag_of_word_dic)
    
    num_labels = 2
    Embedding_dim = 200
    model_2 = BoWClassifier(num_labels, Embedding_dim, size)
    
    '''
    with torch.no_grad():
        bow_vector = create_bag_word(text[1], size, bag_of_word_dic)
        log_probs = model_1(bow_vector)
        if log_probs[0, 1].item() > log_probs[0,0].item():
            print(1)
        else:
            print(0)
        #print(log_probs[0,1].item())
    '''
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_2.parameters(), lr = 0.1)
    # model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
    
    for epoch in range(10):
        x = [i for i in range(len(label))]
        shuffle(x)
        for t in range(len(label)):
            i = x[t]
            model_2.zero_grad()
            
            first_input = create_bag_word(text[i], size, bag_of_word_dic)
            # print(first_input)
            context_idxs = torch.tensor(first_input, dtype=torch.long)
            '''
            # print(len(context_idxs))
            embedding = nn.Embedding(size, Embedding_dim)
            # print(context_idxs)
            embeds = embedding(context_idxs).view(1, 1, -1)
            # print(embeds)
            average_pool = nn.AvgPool1d(len(embeds), len(context_idxs))
            final_input = average_pool(embeds).view((1, -1))
            '''
            
            # print(final_input.shape)
            
            # bow_vec = create_bag_word(text[i], size, bag_of_word_dic)
            target = label[i]
            # print(len(context_idxs))
            if len(context_idxs) == 0:
                continue
            log_probs = model_2(context_idxs)
            
            # print(log_probs.shape)
            # print(int(target))
            target = make_target(target)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            
        print(epoch)
    torch.save(model_2.state_dict(), "/Users/xietiany/Documents/NLP/models/NN_Embedding_model")
    # label_test, sentences_test, text_test = pre_process('data/test.txt')
    
    # with torch.no_grad():
    #     acc_count = 0
    #     for i in range(len(label_test)):
            
    #         first_input = create_bag_word(text_test[i], size, bag_of_word_dic)
    #         # print(len(first_input))
    #         # print(first_input)
    #         context_idxs = torch.tensor(first_input, dtype=torch.long)
    #         '''
    #         # print(len(context_idxs))
    #         embedding = nn.Embedding(size, Embedding_dim)
    #         # print(context_idxs)
    #         embeds = embedding(context_idxs).view(1, 1, -1)
    #         # print(embeds)
    #         average_pool = nn.AvgPool1d(len(embeds), len(context_idxs))
    #         final_input = average_pool(embeds).view((1, -1))
    #         '''
            
    #         log_probs = model_2(context_idxs)
            
    #         if log_probs[0, 1].item() > log_probs[0,0].item():
    #             pred = 1
    #         else:
    #             pred = 0
                
    #         if pred == label_test[i]:
    #             acc_count += 1
            
    #         if i % 1000 == 0:
    #             print(i)
    
    #     print(acc_count / len(label_test))
    #     #outfile.close() 
        
    # sentences_test, text_test = pre_process_unlabel('data/unlabelled.txt')
    
    # with torch.no_grad():
    #     outfile = open("P2_unlabelled.output", "w")
    #     for i in range(len(label_test)):
    #         first_input = create_bag_word(text_test[i], size, bag_of_word_dic)
    #         # print(len(first_input))
    #         # print(first_input)
    #         context_idxs = torch.tensor(first_input, dtype=torch.long)
    #         log_probs = model_2(context_idxs)
            
    #         if log_probs[0, 1].item() > log_probs[0,0].item():
    #             pred = 1
    #         else:
    #             pred = 0
            
    #         outfile.write(str(pred) + '  ' + text_test[i])
            
    #     outfile.close()           
            
    #######----------------------------------------Problem 2 starts here-------------------------------------------
    
def pre_process_unlabel(file):
    text = []
    with open(file, 'r') as f:
        for line in f:
            text.append(line)
    word = []
    for line in text:
        line = "".join(c for c in line if c not in string.punctuation)
        line = [x.strip(string.punctuation) for x in line.split()]
        line = [x.lower() for x in line]
        word.append(line)

    words = []
    for i in range(len(word)):
        for j in range(len(word[i])):
            words.append(word[i][j])
    
    return words, text

def pre_process(file):
    label = []
    text = []
    with open(file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter = ",")
        next(rows)
        for row in rows:
            if int(row[0]) >= 0.5:
                label.append(1)
            else:
                label.append(0)
            text.append(row[2])
    word = []
    for line in text:
        line = "".join(c for c in line if c not in string.punctuation)
        line = [x.strip(string.punctuation) for x in line.split()]
        line = [x.lower() for x in line]
        word.append(line)

    words = []
    for i in range(len(word)):
        for j in range(len(word[i])):
            words.append(word[i][j])
    
    return label, words, text

def create_dic(sens):
    dic = {}
    ind = 0
    for word in sens:
    # print(sen)
        if word not in dic:
            dic[word] = ind
            ind += 1
    return dic, ind

def create_bag_word(line, size, bag_of_word_dic):
    vec = []
    line = "".join(c for c in line if c not in string.punctuation)
    line = [x.strip(string.punctuation) for x in line.split()]
    line = [x.lower() for x in line]
    for word in line:
        if word in bag_of_word_dic:
            vec.append([bag_of_word_dic[word]])
        else:
            vec.append([0])
        
    return vec

def make_target(label):
    return torch.LongTensor([label])    
    '''
    if label == 1:
        return torch.tensor([[0, 1]])
    else:
        return torch.tensor([[1, 0]])
    '''

def bag_words(sen, dic, ind):
    res = np.zeros((1, ind))
    for word in sen:
        if word in dic:
            res[:, dic[word]] = 1
    return res

if __name__== "__main__":
    main()