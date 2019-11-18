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
    def __init__(self, num_labels, embedding_dim, hidden_dim, weight):
        super(BoWClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(weight, freeze = False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
                nn.Linear(embedding_dim, num_labels),
        )
        self.hidden = self.init_hidden()
        # self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
        
    def forward(self, bow_vec):
        embeds = self.embedding(bow_vec).view(1, 1, -1)
        average_pool = nn.AvgPool1d(len(embeds), len(bow_vec))
        final_input = average_pool(embeds).view((1, 1, -1))
        lstm_out, self.hidden = self.lstm(final_input, self.hidden)
        # print(lstm_out.view(1,-1))
        # print(lstm_out.view(1,-1).shape)
        # torch.Size([1, 100])
        linear_input = lstm_out.view(1,-1)
        return torch.sigmoid(self.linear(linear_input))

def main():
    #here is the Problem one model/ # word_to_ix is bag_of_word_dic
    label, sentences, text = pre_process('data/clean_data_1.csv')
    bag_of_word_dic, size, embedding_vec = read_embedding('data/glove.6B.50d.txt')
    
    # print(embedding_vec)
    
    weight = torch.FloatTensor(embedding_vec)
    # print(weight.shape[1])
    # print(sentences)
    # print(bag_of_word_dic)
    
    num_labels = 2
    hidden_dim = 50
    Embedding_dim = weight.shape[1]
    model_2 = BoWClassifier(num_labels, Embedding_dim, hidden_dim, weight)
    
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
    
    for epoch in range(2):
        x = [i for i in range(len(label))]
        shuffle(x)
        for t in range(len(label)):
            i = x[t]
            model_2.zero_grad()
            model_2.hidden = model_2.init_hidden()
            
            first_input = create_bag_word(text[i], size, bag_of_word_dic)
            #print(first_input)
            #print(first_input)
            context_idxs = torch.tensor(first_input, dtype=torch.long)
            #print(context_idxs)
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
            if len(context_idxs) == 0:
                continue
            log_probs = model_2(context_idxs)
            # print(log_probs.size())
            # print(log_probs.shape)
            # print(int(target))
            target = make_target(target)
            loss = loss_function(log_probs, target)
            
            # print('here')
            loss.backward()
            optimizer.step()
            
        print(epoch)
    torch.save(model_2.state_dict(), "/Users/xietiany/Documents/NLP/models/LSTM_model")
    # label_test, sentences_test, text_test = pre_process('data/test.txt')
    
    # with torch.no_grad():
    #     acc_count = 0
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
                
    #         if pred == label_test[i]:
    #             acc_count += 1
            
    #         if i % 1000 == 0:
    #             print(i)
    
    #     print(acc_count / len(label_test))
    #     # outfile.close() 
        
    # sentences_test, text_test = pre_process_unlabel('data/unlabelled.txt')
    
    # with torch.no_grad():
    #     outfile = open("P4_unlabelled.output", "w")
    #     for i in range(len(label_test)):
    #         first_input = create_bag_word(text_test[i], size, bag_of_word_dic)
    #         context_idxs = torch.tensor(first_input, dtype=torch.long)

    #         log_probs = model_2(context_idxs)
            
    #         if log_probs[0, 1].item() > log_probs[0,0].item():
    #             pred = 1
    #         else:
    #             pred = 0
            
    #         outfile.write(str(pred) + '  ' + text_test[i])
            
    #     # print(acc_count / len(label_test))
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

def read_embedding(file):
    dic = {}
    embedding_vec = []
    with open(file, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            line = line.split()
            dic[line[0]] = count
            count += 1
            vec = [float(i) for i in line[1:]]
            embedding_vec.append(vec)
            #print(len(line[1:]))
    return dic, count, embedding_vec
        #job_titles = [line.decode('utf-8').strip() for line in f.readlines()]

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