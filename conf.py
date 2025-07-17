# 2023年月11日14时17分12秒
from transformers import BertTokenizer
import os
import argparse
import configparser as ConfigParser
from optparse import OptionParser
import pdb

# test_dataset = 'wav2vec_version/wav2vec_transcripts'


test_dataset = 'wav2vec_transcripts'
# test_list = 'ADReSSo_list'
test_list = 'test_list'
classification_task = 'BERT_tunning'
batch_size = 3  # should be global variable
drop_rate = 0.3
epochs = 8
l2 = 0.0001
class_num = 2
SEED = 1024
output_hidden_states = False

train_data_folder='ASR_Results_wav2vec/after_tunning/adresso_train'
test_data_folder='ASR_Results_wav2vec/after_tunning/adresso_test'

label_dict_path='nfoldsplits/label_dict.npy'

wav2vec_feats_folder='Feature_Extractor/feature/ADReSSo'
text_feats_folder = 'Feature_Extractor/feature/bert_feats'
text_scritps_folder = 'Feature_Extractor/feature/scritpss'
time_align_dict_folder = ''


def make_path(options):
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(options.pre_trained_model, do_lower_case=True)
    root_folder = "saved_models"
    model_save_folder = os.path.join(root_folder, test_dataset, "fine_tune_models")
    results_save_folder = os.path.join(root_folder ,test_dataset, "results")

    if os.path.exists(model_save_folder) == False:
        os.makedirs(model_save_folder)
    if os.path.exists(results_save_folder) == False:
        os.makedirs(results_save_folder)

    # data_list_folder = os.path.join("../data/ac1yp/data/cookie_theft/lists/", options.train_task, "list")
    data_list_folder='nfoldsplits'
    return model_save_folder, results_save_folder, data_list_folder,tokenizer

