import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, f1_score, confusion_matrix
import sys
import pdb
from collections import Counter
test_list_path = 'nfoldsplits/test1.csv'


def result_save(test_list, pred_label_list):
    pred_label_dict = {}
    for test_id, pred_label in zip(test_list, pred_label_list):
        pred_label_dict[test_id[:-1]] = pred_label
    np.save(output_dict_path, pred_label_dict)



pred_label_folder = "saved_models/wav2vec_transcripts/results/pred_labels/123"
all_dirs = os.listdir(pred_label_folder)  # ['0']


test_list = open(test_list_path).readlines()  # 读取测试集的说话人
test_label_list = []

for test_id in test_list:

    name = test_id[:-1]
    name, label = name.split(',')
    name=os.path.basename(name).replace('.txt','')

    # print(name[-2:].replace('_',''))
    if int(name[-2:].replace('_',''))==1:
        if int(name.rsplit("_",1)[0].replace('adrsdt',''))==49:
            pass
        label=int(label)
        test_label_list.append(label)
    else:
        pass
conter=Counter(test_label_list)
print(conter[1])
print(conter[0])
# for test_id in test_list:
#     name = test_id[:-1]
#     name, label = name.split(',')
#     label=int(label)
#     test_label_list.append(label)


pred_label_matrix = []
f_score_list = []
accuracy_list = []
Recll_list = []
precision_list = []
Fold=1
for Fold in range(10):


    output_dict_path = os.path.join(
        'saved_models/final/test_results/',
        str(Fold) + 'test_pred_label_dict.npy')
    # sub_dirs_path = os.path.join()
    pred_label_path = os.path.join(pred_label_folder, str(Fold), 'bert-base-uncased_pred_labels.npy')
    # pred_label_path = os.path.join(pred_label_folder, str(Fold), 'test2.npy')
    pred_label = np.load(pred_label_path)
    conter1=Counter(pred_label)
    print(conter1[1])
    print(conter1[0])
    if len(pred_label)!= len(test_label_list):
        test_label_list=test_label_list[:len(pred_label)]
    conter2=Counter(test_label_list)
    print(conter2[1])
    print(conter2[0])
    accuracy = accuracy_score(test_label_list, pred_label)
    print(accuracy)
    pre, rec, f_1, U = precision_recall_fscore_support(test_label_list, pred_label)
    print("  Accuracy: {:.2f}\t"
          "  precision: {:.2f}\t"
          "  f1 score: {:.2f}\t"
          "  confusion matrix: {}".format(accuracy_score(test_label_list, pred_label),
                                          precision_score(test_label_list, pred_label, average='weighted'),
                                          f1_score(test_label_list, pred_label, average='weighted'),
                                          confusion_matrix(test_label_list, pred_label).tolist()))
    print('pre:', pre, 'rec:', rec, "f1:", f_1, U)
    print('\t')
    accuracy_list.append(accuracy)
    f_score_list.append(f_1)
    Recll_list.append(rec)
    precision_list.append(pre)

    pred_label_matrix.append(pred_label)

print("  f1 score:\t",f_score_list)
print("  Accuracy:\t",accuracy_list)
print("  precision:\t",precision_list)
print("  Recll:\t",Recll_list)
print("  Accuracy:\t",np.mean(accuracy_list))
print("  f1 score:\t",np.mean(f_score_list,axis=0))
print("  precision:\t",np.mean(precision_list,axis=0))
print("  Recll:\t",np.mean(Recll_list,axis=0))

test_label_list = np.asarray(test_label_list)
pred_label_matrix = np.asarray(pred_label_matrix)

pred_label_list = np.mean(pred_label_matrix, axis=0)
# round(x)四舍五入
pred_label_list = np.asarray([round(x) for x in pred_label_list])

# result_save(test_list, pred_label_list)