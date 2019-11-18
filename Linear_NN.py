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

class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        
        self.linear = nn.Sequential(
                nn.Linear(vocab_size, num_labels),
        )
        
    def forward(self, bow_vec):
        return torch.sigmoid(self.linear(bow_vec))

def main():
    #here is the Problem one model/ # word_to_ix is bag_of_word_dic
    label, sentences, text = pre_process('data/clean_data_1.csv')
        
    # print(sentences)
    bag_of_word_dic, size = create_dic(sentences)
    # print(bag_of_word_dic)
    
    num_labels = 2
    model_1 = BoWClassifier(num_labels, size)
    
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
    optimizer = optim.SGD(model_1.parameters(), lr = 0.1)
        
    for epoch in range(3):
        x = [i for i in range(len(label))]
        shuffle(x)
        for t in range(len(label)):
            i = x[t]
            model_1.zero_grad()
            
            bow_vec = create_bag_word(text[i], size, bag_of_word_dic)
            target = label[i]
            # print(bow_vec.shape)
            log_probs = model_1(bow_vec)
            
            # print(log_probs.shape)
            # print(int(target))
            target = make_target(target)
            # print(target.shape)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            
        print(epoch)
        
    torch.save(model_1.state_dict(), "/Users/xietiany/Documents/NLP/models/NN_model")
            
            
    #######----------------------------------------Problem 2 starts here-------------------------------------------

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
    vec = torch.zeros(size)
    line = "".join(c for c in line if c not in string.punctuation)
    line = [x.strip(string.punctuation) for x in line.split()]
    line = [x.lower() for x in line]
    for word in line:
        if word in bag_of_word_dic:
            vec[bag_of_word_dic[word]] = 1
        
    return vec.view(1, -1)

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