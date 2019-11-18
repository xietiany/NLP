#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:02:26 2017

@author: xietiany
"""
import sys
import operator
import csv
second_arg = sys.argv[1]
third_arg = sys.argv[2]


def main(train_file_name, test_file_name):
    tag_word_dic = {}
    tag_tag_dic = {}
    tag_tag_dic['white_space'] = {}
    total_count_tag_word_dic = {}
    total_count_tag_tag_dic = {}
    baseline_dic = {}
    total_count_tag_tag_dic['white_space'] = 0
    #training data to record data
   
    #use training data to create dictionary
    #1.tag_word_dic: is a dictionary of count of word based on given tag, for example, race/given_tag_is_Noun = 1000,
    #race/given_tag_is_Verb = 200
    #2.tag_tag_dic: is a dictionary of count of tag based on previous tag, for example Noun/Noun = 100, Verb/Noun = 200
    #3.total_count_tag_word_dic: is a dictionary that count the total number of each tag given in the training data
    #for example, C(Noun) = 54000, C(Verb) = 17000....
    #4.total_count_tag_tag_dic: is similar to total_count_tag_word_dic, but used in tag_previous_tag probability counting
    #baseline_dic: is a dictionary count for each word, their tag count. it is used in navie baseline method
    with open(train_file_name, 'r') as f:
        for line in f:
            each_word = line.split()
            for i in range(len(each_word)):
                if len(each_word[i].split('/')) >= 2:
                
                    
                    word = each_word[i].split('/')[0]
                    tag = each_word[i].split('/')[1]
                    
                    #construct baseline_dic
                    if word not in baseline_dic:
                        baseline_dic[word] = {}
                    if tag not in baseline_dic[word]:
                        baseline_dic[word][tag] = 1
                    else:
                        baseline_dic[word][tag] += 1
                    #construct tag_word dictionary
                    #construct total_count_tag_word_dic
                    if tag not in tag_word_dic:
                        tag_word_dic[tag] = {}
                        total_count_tag_word_dic[tag] = 1
                    if word not in tag_word_dic[tag]:
                        tag_word_dic[tag][word] = 1
                        total_count_tag_word_dic[tag] += 1
                    else:
                        tag_word_dic[tag][word] += 1
                        total_count_tag_word_dic[tag] += 1
                    #construct tag_tag dictionary
                    #construct total_count_tag_tag_dic
                    if i == 0:
                        if tag not in tag_tag_dic['white_space']:
                            tag_tag_dic['white_space'][tag] = 1
                            total_count_tag_tag_dic['white_space'] += 1
                        else:
                            tag_tag_dic['white_space'][tag] += 1
                            total_count_tag_tag_dic['white_space'] += 1
                    if i != 0:
                        if len(each_word[i - 1].split('/')) >= 2:
                            previous_tag = each_word[i - 1].split('/')[1]
                            if previous_tag not in tag_tag_dic:
                                tag_tag_dic[previous_tag] = {}
                                total_count_tag_tag_dic[previous_tag] = 1
                            if tag not in tag_tag_dic[previous_tag]:
                                tag_tag_dic[previous_tag][tag] = 1
                                total_count_tag_tag_dic[previous_tag] += 1
                            else:
                                tag_tag_dic[previous_tag][tag] += 1
                                total_count_tag_tag_dic[previous_tag] += 1

    # print(baseline_dic['race'])


    #test data to analyze
    #--------------------------------------read test file-----------------------
    #---------------------------------------------------------------------------
    # with open("data/train.csv", "r") as csv_file:
    #     rows = csv.reader(csv_file, delimiter = ",")
    #     count = 0
    #     for row in rows:
    #         if (count >= 1 and count <= total_size):
    #             target_target = (row[0], row[1], row[2])
    #             insert_tasks(conn, target_query, target_target)
    #             # insert_tasks(conn, other_target_query, other_target_target)
    #             # insert_tasks(conn, hate_index_query, hate_index_target)
    #         count += 1

    
    word_vector = []
    # real_tag = []
    with open(test_file_name, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter= ",")
        for row in rows:
            word_vector.append(list(row[2].split()))
        # for line in f:
        #     each_word = line.split()
        #     temp_word = []
        #     temp_tag = []
        #     for i in range(len(each_word)):
        #         temp_word.append(each_word[i].split('/')[0])
        #         temp_tag.append(each_word[i].split('/')[1])
        #     word_vector.append(temp_word)
        #     real_tag.append(temp_tag)
    
    predict_vector = []
    #----------------------------predict each word tag for each sentence---------------------
    #----------------------------------------------------------------------------------------
    for i in range(len(word_vector)):
        
        sentence = word_vector[i]
        score = []
        back_pointer = []
        temp_score = []
        temp_back_pointer = []
        #--------------------------------initialize first score row and backpointer row--------------------
        #--------------------------------------------------------------------------------------------------
        for key in tag_word_dic:
            if sentence[0] not in tag_word_dic[key]:
                P_word_given_tag = 1 / (total_count_tag_word_dic['NN'] + 1)
            else:
                P_word_given_tag = tag_word_dic[key][sentence[0]] / (total_count_tag_word_dic[key] + 1)
            if key not in tag_tag_dic['white_space']:
                P_tag_given_tag = 1 / (total_count_tag_tag_dic['white_space'] + 1)
            else:
                P_tag_given_tag = tag_tag_dic['white_space'][key] / (total_count_tag_tag_dic['white_space'] + 1)
            temp_score.append(P_word_given_tag * P_tag_given_tag)
            temp_back_pointer.append('')
        score.append(temp_score)
        back_pointer.append(temp_back_pointer)
    
        #-----------------------------------dynamic programming, Viterbi algorithm--------------------------
        #---------------------------------------------------------------------------------------------------
        for word in sentence[1:]:
            temp_score = []
            temp_back_pointer = []
            for tag in tag_word_dic:
                if word not in tag_word_dic[tag]:
                    P_word_given_tag = 1 / (total_count_tag_word_dic['NN'] + 1)
                else:
                    P_word_given_tag = tag_word_dic[tag][word] / (total_count_tag_word_dic[tag] + 1)
    
                temp_value = 0
                temp_index = ''
                count_iter = 0
    
                for tag_2 in tag_word_dic:
                    if tag not in tag_tag_dic[tag_2]:
                        P_tag_given_tag = 1 / (total_count_tag_word_dic['NN'] + 1)
                    else:
                        P_tag_given_tag = tag_tag_dic[tag_2][tag] / (total_count_tag_word_dic[tag_2] + 1)
                    if score[-1][count_iter] * P_tag_given_tag > temp_value:
                        temp_value = score[-1][count_iter] * P_tag_given_tag
                        temp_index = tag_2
                    count_iter += 1
    
                temp_score.append(P_word_given_tag * temp_value)
                temp_back_pointer.append(temp_index)
            score.append(temp_score)
            back_pointer.append(temp_back_pointer)
    
        #determine index
        #create a key_number dictionary, create a key and their position one to one relationship
        #so we can trace back by backpointer which provide in which position, and finally get our tag
        #--------------------------------------------------------------------------------------------
        key_number = {}
        count = 0
        for key in tag_word_dic:
            key_number[key] = count
            count += 1
    
        number_key = {}
        count = 0
        for key in tag_word_dic:
            number_key[count] = key
            count += 1
    
        #Use backpointer to trace back and get our tag prediction in a opposite order
        #----------------------------------------------------------------------------
        index_sequence = []
        m = max(score[-1])
        index = [i for i, j in enumerate(score[-1]) if j == m][0]
        predict_tag = number_key[index]
        index_sequence.append(predict_tag)
    
        for w in range(len(back_pointer)-2, -1 , -1):
            temp_index = key_number[predict_tag]
            predict_tag = back_pointer[w+1][temp_index]
            index_sequence.append(predict_tag)
            #index = back_pointer[w+1].index('')
        
        #reverse the order, since we get the tag from the last word to the beginning word
        #--------------------------------------------------------------------------------
        true_order = []
        for i in reversed(index_sequence):
            true_order.append(i)
        predict_vector.append(true_order)
        #print(true_order)
    '''
    print(back_pointer[-1][10])
    print(key_number['NP'])
    print(back_pointer[-2][0])
    print(key_number['IN'])
    print(back_pointer[-3][9])
    print(key_number['NNS'])
    print(back_pointer[-4][3])
    print(key_number['NN'])
    print(back_pointer[-5][8])
    print(key_number['WRB'])
    print(back_pointer[-6][30])
    '''
    outfile = open("results/POS.test.out", "w")
    correct_count = 0
    total_count = 0
    for i in range(len(predict_vector)):
        for j in range(len(predict_vector[i])):
            outfile.write(word_vector[i][j] +  "/" + predict_vector[i][j] + " ")
        outfile.write('\n')
    # accuracy = correct_count / total_count
    # print("Viterbi algorithm accuracy is: ", accuracy)
    #-------------------------------------------------------------------------------------------------------------------
    #--------------------------------------Naive method to predict baseline accuracy------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    # baseline_method_tag_prediction = []
    # for i in range(len(word_vector)):
    #     temp_tag = []
    #     #print('here')
    #     for j in range(len(word_vector[i])):
    #         if word_vector[i][j] in baseline_dic:
    #             temp_tag.append(max(baseline_dic[word_vector[i][j]], key = baseline_dic[word_vector[i][j]].get))
    #         else:
    #             temp_tag.append('NN')
    #     baseline_method_tag_prediction.append(temp_tag)
    
    # baseline_method_count = 0
    # baseline_method_total_count = 0
    # for i in range(len(baseline_method_tag_prediction)):
    #     for j in range(len(baseline_method_tag_prediction[i])):
    #         if baseline_method_tag_prediction[i][j] == real_tag[i][j]:
    #             baseline_method_count += 1
    #             baseline_method_total_count += 1
    #         else:
    #             baseline_method_total_count += 1
    # accuracy2 = baseline_method_count / baseline_method_total_count
    # print("Baseline method naive method accuracy is: ", accuracy2)
    
if __name__ == '__main__':
    main(second_arg, third_arg)