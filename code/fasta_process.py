#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from tqdm import tqdm
# 在fasta_process.py开头添加
import os
os.makedirs('./data', exist_ok=True)




def read_file(filepath):
    with open(filepath) as fp:
        content=fp.read();
    return content





file = read_file('../data/arg_v5.fasta').split('>')
len(file)





print(type(file))





print(type(file))





# file[0] is null
del file[0]




data = pd.DataFrame()
anti_label = {}
mech_label = {}
type_label = {}
line = file[0].split('/')
# print(file[0])
# print(len(line))
for f in tqdm(file):
    line = f.split('|')
    r6 = line[6].split('\n')
 # 获取序列信息  
    seq = ''
    for i in range(1, len(r6)):
        seq += r6[i]
#     统计不同类别的数量
    if line[3] not in anti_label:
        anti_label[line[3]] = 0
    if line[5] not in mech_label:
        mech_label[line[5]] = 0
    if r6[0] not in type_label:
        type_label[r6[0]] = 0
    anti_label[line[3]] += 1
    mech_label[line[5]] += 1
    type_label[r6[0]] += 1
    data = data._append({'id':line[0], 'antibiotic':line[3],'arg':line[4],'mechanism':line[5],'type': r6[0], 'seq': seq}, ignore_index=True)
  
data


data = pd.DataFrame()
anti_label = {}
mech_label = {}
type_label = {}
for f in tqdm(file):
    line = f.split('|')
    r6 = line[6].split('\n')
    seq = ''
    for i in range(1, len(r6)):
        seq += r6[i]
    if line[3] not in anti_label:
        anti_label[line[3]] = 0
    if line[5] not in mech_label:
        mech_label[line[5]] = 0
    if r6[0] not in type_label:
        type_label[r6[0]] = 0
    anti_label[line[3]] += 1
    mech_label[line[5]] += 1
    type_label[r6[0]] += 1
    data = data._append({'id':line[0], 'antibiotic':line[3],'arg':line[4],'mechanism':line[5],'type': r6[0], 'seq': seq}, ignore_index=True)
data





data['seq'][1]





data['seq'][1]
data.to_csv("./data/res.csv")





len(anti_label)





anti_label





sorted(anti_label.items(), key=lambda x: x[1], reverse=True)





mech_label





type_label




data





maxlen = 0
for index, row in data.iterrows():
    l = len(row['seq'])
    maxlen = max(l, maxlen)
maxlen





word_map = {}
used_anti_label = {'beta_lactam': 0, 'bacitracin': 1,'multidrug': 2,'macrolide-lincosamide-streptogramin': 3,'aminoglycoside': 4,
                   'polymyxin': 5,'chloramphenicol': 6, 'tetracycline': 7,'fosfomycin': 8,'glycopeptide': 9,'quinolone': 10,
                   'trimethoprim': 11, 'sulfonamide': 12, 'rifampin': 13,'others': 14}
uesd_mech_label = {'antibiotic target protection': 0,  'antibiotic efflux': 1, 'antibiotic inactivation': 2, 
                   'antibiotic target alteration': 3,'antibiotic target replacement': 4, 'others': 5}
def seq2Onehot(row):
    seq = row['seq']
    seq_mat = []
    for w in seq:
        if w not in word_map:
            word_map[w] = len(word_map)
        one_hot = [0] * 23
        one_hot[word_map[w]] = 1
        seq_mat.append(one_hot)
    
    # zero-padding
    for i in range(len(seq_mat), maxlen):
        one_hot = [0] * 23
        seq_mat.append(one_hot)
    
    row['seq_map'] = seq_mat
    
    if row['antibiotic'] not in used_anti_label:
        row['anti_label'] = 14
    else:
        row['anti_label'] = used_anti_label[row['antibiotic']]
    
    if row['mechanism'] not in uesd_mech_label:
        row['mech_label'] = 5
    else:
        row['mech_label'] = uesd_mech_label[row['mechanism']]
    
    row['type_label'] = int(row['type'])
    return row
data = data.apply(seq2Onehot, axis=1)
data





data['seq_map'][25]





len(data['seq_map'][1])





data['type_label'].sum()




from collections import Counter




Counter(data['anti_label'])




Counter(data['mech_label'])



Counter(data['type_label'])




data.to_csv('./data/arg_v5_processed_withoutseq.csv')



data.to_pickle('./data/arg_v5_processed.pickle')



data.head(300).to_pickle('./data/arg_v5_processed_mini.pickle')





a = np.array([[1,9,3],[4,5,6]])
a



np.argmax(a[0])

