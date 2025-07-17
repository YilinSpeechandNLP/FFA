# 2023年月19日16时32分18秒
import csv
import os

import numpy as np

def sort_by_number(file_name):
    # 提取文件名中的数字部分
    file_name1= file_name.rsplit('_', 1)[0]
    file_name2=file_name.rsplit('_', 1)[1]
    number1 = int(''.join(filter(str.isdigit, file_name1)))
    number2 = int(''.join(filter(str.isdigit, file_name2)))
    return (number1,number2)
# str1= 'cd_sdsdsd15_11'
# print(str1.rsplit("_",1)[0])
# print(str1.rsplit("_",1)[1])
# print(sort_by_number(str1))
# print(str1.split('_')[0])
def nfoldsplits(wav_path,csv_path,output_path,script_path,wav_path2=None):
    with open(csv_path) as f:
        items = list(csv.reader(f))
        entir = []
        entir_dict = {}
        for i in items:
            b = os.path.basename(i[0]).replace('.log','')
            entir_dict[b] = i[1]
            entir.append(b)
    wav_list = os.listdir(wav_path)
    wav_list = sorted(wav_list, key=sort_by_number)
    fold = []
    for i in wav_list:
        ii = i.rsplit('_', 1)[0]
        # ii = i.split('_')[0]

        if ii in entir:
            file_path = os.path.join(script_path, i)
            label = entir_dict[ii]
            b = [file_path.replace('../','').replace('.wav','.txt'), label]
            fold.append(b)
    if wav_path2 is not None:
        wav_list1 = os.listdir(wav_path2)
        wav_list1 = sorted(wav_list1, key=sort_by_number)
        for i in wav_list1:
            ii = i.rsplit('_', 1)[0]
            # ii = i.split('_')[0]

            if ii in entir:
                file_path = os.path.join(script_path, i)
                label = entir_dict[ii]
                b = [file_path.replace('../', '').replace('.wav', '.txt'), label]
                fold.append(b)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fold)

script_path='../Feature_Extractor/feature/scritps'
wav_path='../ADReSSo/audio/train-segment/cn'
wav_path2='../ADReSSo/audio/train-segment/ad'
test_path='../ADReSSo/audio/test-dist-segment'
nfoldsplits(test_path,'test.csv','test1.csv',script_path)
# for i in range(10):
#     csv_path='train_{}.csv'.format(i)
#     output_path='train{}_{}.csv'.format(i+1,i)
#     nfoldsplits(wav_path,csv_path,output_path,script_path,wav_path2)
# for i in range(10):
#     csv_path='val_{}.csv'.format(i)
#     output_path='val{}_{}.csv'.format(i+1,i)
#     nfoldsplits(wav_path,csv_path,output_path,script_path,wav_path2)
# with open('test.csv') as f:
#     items=list(csv.reader(f))
# print(items)
# b=[]
# for i in range(len(items)):
#     ab=items[i][0].replace('Data/Scripts_Continuous\\','')
#     label=items[i][1]
#     b.append([ab,label])
# print(b)
#
dict=np.load('label_dict.npy',allow_pickle=True).item()
# print(len(dict))
# for i in range(len(b)):
#     name=b[i][0]
#     label=int(b[i][1])
#     dict[name]=label
print(dict)
# print(len(dict))
# np.save('label_dict.npy',dict)