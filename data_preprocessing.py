# 2023年月11日14时07分02秒
import os
import numpy as np
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re
import torch
from conf import *
from sklearn.preprocessing import StandardScaler
# from Models import  TransformerBlock, TokenAndPositionEmbedding
from Models import TransformerBlock, TokenAndPositionEmbedding
from keras.utils.np_utils import to_categorical
import pdb
from keras import layers
import random
# import tensorflow as tf
from sklearn.preprocessing import normalize


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\+", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\//", "", string)
    return string.strip().lower()


def data_load(data_list, data_folder, label_dict):
    texts = []
    labels = []
    for item in data_list:
        name = item[:-1]
        label = label_dict[name]
        trans_path = os.path.join(data_folder, name)
        transcript = open(trans_path).readlines()[0]
        transcript = clean_str(transcript)
        texts.append(transcript)
        labels.append(label)
    return labels, texts


def sentence_tokenize(sentences, tokenizer, add_special_tokens):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=add_special_tokens,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            truncation=True,
            max_length=512,  # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=MAX_TRAN_LEN, dtype="long",
                              value=0, truncating="post", padding="post")
    # print('\Done.')

    return input_ids

def sentence_tokenize1(sentences, tokenizer, add_special_tokens):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids_sub = []
    input_ids=[]
    # For every sentence...
    for sent in sentences:
        for ss in sent:
            encoded_sent = tokenizer.encode(
                ss,  # Sentence to encode.
                add_special_tokens=add_special_tokens,  # Add '[CLS]' and '[SEP]'
                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                truncation=True,
                max_length=512,  # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids_sub.append(encoded_sent)

        input_ids.append(input_ids_sub)
        input_ids_sub=[]

    for i in range(len(input_ids)):
        input_ids[i] = pad_sequences(input_ids[i], maxlen=MAX_TRAN_LEN, dtype="long",
                               value=0, truncating="post", padding="post")
    # print('\Done.')

    return input_ids

def sentence_mask(input_ids):
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return attention_masks
def sentence_mask1(input_ids):
    attention_masks_sub = []
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        for ss in sent:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in ss]

            # Store the attention mask for this sentence.
            attention_masks_sub.append(att_mask)
        attention_masks.append(attention_masks_sub)
        attention_masks_sub=[]
    return attention_masks

def wav2vec_load(data_list, feats_folder):
    feats = []
    for item in data_list:
        name = item[:-1]
        feats_path = os.path.join(feats_folder, name + '.npy')
        feat = np.load(feats_path)
        avg_feat = np.mean(feat, axis=0)
        feats.append(avg_feat)
    feats = np.asarray(feats)
    feats = normalize(feats)
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    return feats


def data_load1(data_list, data_folder, feats_folder, label_dict):
    texts = []
    labels = []
    wav2vec_feats = []
    data_len = len(data_list)
    data_list1 = [data_list[i - 1] for i in range(data_len, 0, -1)]
    for i, item in enumerate(data_list):
        name_t = item[:-1]
        name_t, label_t = name_t.split(',')
        label_t = int(label_t)
        name_t = os.path.basename(name_t).replace('.txt', '')

        name_a = data_list1[i][:-1]
        name_a, label_a = name_a.split(',')
        label_a = int(label_a)
        name_a = os.path.basename(name_a).replace('.txt', '')

        if name_a.startswith('ad_'):
            feats_path = os.path.join(feats_folder, name_a + '.npy')
        else:
            feats_path = os.path.join(feats_folder, name_a + '.wav.npy')
        feat = np.load(feats_path)

        wav2vec_feats.append(feat)
        wav2vec_feats.append(feat)
        if name_t.startswith('ad_'):
            feats_path = os.path.join(feats_folder, name_t + '.npy')
        else:
            feats_path = os.path.join(feats_folder, name_t + '.wav.npy')
        feat = np.load(feats_path)

        wav2vec_feats.append(feat)
        wav2vec_feats.append(feat)

        trans_path = os.path.join(data_folder, name_a + '.txt')
        with open(trans_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            if len(txt) == 0:
                txt = ' '
            transcript = txt[0][:-1]  # get txt information

        trans_path2 = os.path.join(data_folder, name_t + '.txt')
        with open(trans_path2, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            if len(txt) == 0:
                txt = ' '
            transcript2 = txt[0][:-1]  # get txt information

        texts.append(transcript)
        texts.append(transcript2)
        texts.append(transcript)
        texts.append(transcript2)

        label = label_a * 2 + label_a
        labels.append(label)
        label = label_a * 2 + label_t
        labels.append(label)
        label = label_t * 2 + label_a
        labels.append(label)
        label = label_t * 2 + label_t
        labels.append(label)

    return labels, texts, wav2vec_feats


def data_load2(data_list, data_folder, feats_folder, label_dict):
    texts = []
    labels = []
    wav2vec_feats = []
    for i, item in enumerate(data_list):
        name_t = item[:-1]
        name_t, label_t = name_t.split(',')
        name_t = os.path.basename(name_t).replace('.txt', '')
        if name_t.startswith('ad_'):
            name_tt = name_t.replace('ad_', '')
        else:
            name_tt = name_t.replace('cn_', '')

        label_t = int(label_t)
        a = int(name_tt[5:8])
        if 23 < a < 145 or 186 < a < 254:
            while True:
                aa = random.randint(930, len(data_list) - 1)  # 生成23到254之间的随机整数

                if aa != a:
                    break

        else:
            while True:
                aa = random.randint(0, 910)  # 生成1到316之间的随机整数

                if aa != a:
                    break
        name_a = data_list[aa][:-1]
        name_a, label_a = name_a.split(',')
        name_a = os.path.basename(name_a).replace('.txt', '')

        trans_path = os.path.join(data_folder, name_t + '.txt')
        with open(trans_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            if len(txt) == 0:
                txt = ' '
            transcript = txt[0][:-1]  # get txt information

        texts.append(transcript)
        if name_a.startswith('ad_'):
            feats_path = os.path.join(feats_folder, name_a + '.npy')
        else:
            feats_path = os.path.join(feats_folder, name_a + '.wav.npy')
        feat = np.load(feats_path)

        wav2vec_feats.append(feat)
        labels.append(label_t)

    return labels, texts, wav2vec_feats


def aux1_fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer):
    labels, texts, wav2vec_feats = data_load1(data_list, text_sub_folder, wav2vec_sub_folder, label_dict)
    labels = labels[:int(len(labels) / 2 + 10)]
    texts = texts[:int(len(texts) / 2 + 10)]
    wav2vec_feats = wav2vec_feats[:int(len(wav2vec_feats) / 2 + 10)]
    inputs_id = sentence_tokenize(texts, tokenizer, add_special_tokens=True)
    attention_masks = sentence_mask(inputs_id)
    inputs_id = torch.Tensor(inputs_id).to(torch.int64)
    attention_masks = torch.Tensor(attention_masks).to(torch.int64)

    # texts_feats, t_mask = pad_input(texts_feats, lenght=40)
    wav2vec_feats, a_mask = pad_input(wav2vec_feats, lenght=550)


    wav2vec_feats = torch.stack([torch.from_numpy(arr) for arr in wav2vec_feats], dim=0).to(torch.float32)
    a_mask = torch.stack([torch.from_numpy(arr) for arr in a_mask], dim=0)
    labels = torch.Tensor(labels).to(torch.int64)

    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, a_mask, labels)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader


def aux2_fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer):
    labels, texts, wav2vec_feats = data_load2(data_list, text_sub_folder, wav2vec_sub_folder, label_dict)

    inputs_id = sentence_tokenize(texts, tokenizer, add_special_tokens=True)
    attention_masks = sentence_mask(inputs_id)
    inputs_id = torch.Tensor(inputs_id).to(torch.int64)
    attention_masks = torch.Tensor(attention_masks).to(torch.int64)

    # texts_feats, t_mask = pad_input(texts_feats, lenght=40)
    wav2vec_feats, a_mask = pad_input(wav2vec_feats, lenght=550)
    # texts_feats = torch.stack([torch.from_numpy(arr) for arr in texts_feats], dim=0).to(torch.float32)
    # t_mask = torch.stack([torch.from_numpy(arr) for arr in t_mask], dim=0)
    wav2vec_feats = torch.stack([torch.from_numpy(arr) for arr in wav2vec_feats], dim=0).to(torch.float32)
    a_mask = torch.stack([torch.from_numpy(arr) for arr in a_mask], dim=0)
    labels = torch.Tensor(labels).to(torch.int64)

    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, a_mask, labels)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader


def data_load3(data_list, text_sub_folder, wav2vec_sub_folder, label_dict):
    texts = []
    labels = []
    wav2vec_feats = []
    # wlength = []
    # tlength = []
    for i, item in enumerate(data_list):
        name = item[:-1]
        name, label = name.split(',')
        label = int(label)
        name = os.path.basename(name).replace('.txt', '')
        tex_name = name.rsplit('_', 1)[0]
        if name.startswith('ad_'):
            # tex_name = name.replace('ad_', '')
            feats_path = os.path.join(wav2vec_sub_folder, name + '.npy')
        else:
            # tex_name = name.replace('cn_', '')
            feats_path = os.path.join(wav2vec_sub_folder, name + '.wav.npy')
        # label = label_dict[name]

        wav_feat = np.load(feats_path)
        # wlength.append(wav_feat.shape[1])
        # avg_feat = np.mean(wav_feat, axis=0)
        wav2vec_feats.append(wav_feat)

        # trans_path = os.path.join(text_sub_folder, name + '_dys.txt')
        trans_path = os.path.join(text_sub_folder, name + '.txt')
        with open(trans_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            if len(txt) == 0:
                txt = ' '
            transcript = txt[0][:-1]  # get txt information

        texts.append(transcript)
        labels.append(label)
    return labels, texts, wav2vec_feats


def pad_input(xx, lenght, pad_value=0):
    feats = []
    attention_mask = []
    for x in xx:
        if len(x.shape) == 3:
            x = x.mean(axis=0)
        t = x.shape[0]
        # mask = torch.zeros(lenght)
        mask = np.zeros(lenght)

        if lenght > t:
            x = np.pad(x, ((0, lenght - t), (0, 0)), 'constant', constant_values=(pad_value, pad_value))
            mask[-(lenght - t):] = 1

        else:
            x = x[:lenght, :]
        # x = torch.from_numpy(x)
        mask = mask.astype(bool)
        # mask = mask.eq(1)
        feats.append(x)
        attention_mask.append(mask)
    return feats, attention_mask


def pad_input_2D(xx, lenght, pad_value=0):
    feats = []
    attention_mask = []
    for xxx in xx:
        subfeats = []
        subattention_mask = []
        for x in xxx:
            if len(x.shape) == 3:
                x = x.mean(axis=0)
            t = x.shape[0]
            # mask = torch.zeros(lenght)
            mask = np.zeros(lenght)
            if lenght > t:
                x = np.pad(x, ((0, lenght - t), (0, 0)), 'constant', constant_values=(pad_value, pad_value))
                mask[-(lenght - t):] = 1
            else:
                x = x[:lenght, :]
            # x = torch.from_numpy(x)
            mask = mask.astype(bool)
            # mask = mask.eq(1)
            subfeats.append(x)
            subattention_mask.append(mask)
        feats.append(subfeats)
        attention_mask.append(subattention_mask)

    return feats, attention_mask


def data_load4(data_list, text_sub_folder, wav2vec_sub_folder, label_dict):
    texts = []
    texts_sub = []
    labels = []
    labels_sub = []
    wav2vec_feats = []
    wav2vec_subfeats = []
    # wlength = []
    # tlength = []
    for i, item in enumerate(data_list):
        name = item[:-1]
        name, label = name.split(',')
        label = int(label)
        name = os.path.basename(name).replace('.txt', '')

        if int(name[-2:].replace('_', '')) == 1:
            if len(texts_sub) != 0:
                texts.append(texts_sub)
                wav2vec_feats.append(wav2vec_subfeats)
                labels.append(labels_sub)
                texts_sub = []
                wav2vec_subfeats = []
                labels_sub = []
            if name.startswith('ad_'):
                feats_path = os.path.join(wav2vec_sub_folder, name + '.npy')
            else:
                feats_path = os.path.join(wav2vec_sub_folder, name + '.wav.npy')
            wav_feat = np.load(feats_path)
            # wlength.append(wav_feat.shape[1])
            wav2vec_subfeats.append(wav_feat)

            # trans_path = os.path.join(text_sub_folder, name + '_dys.txt')
            trans_path = os.path.join(text_sub_folder, name + '.txt')
            with open(trans_path, 'r', encoding='utf-8') as f:
                txt = f.readlines()
                if len(txt) == 0:
                    txt = ' '
                transcript = txt[0][:-1]  # get txt information
            # tlength.append(trans_feat.shape[0])
            texts_sub.append(transcript)

            labels_sub.append(label)
        else:
            if name.startswith('ad_'):
                feats_path = os.path.join(wav2vec_sub_folder, name + '.npy')
            else:
                feats_path = os.path.join(wav2vec_sub_folder, name + '.wav.npy')
            # feats_path = os.path.join(wav2vec_sub_folder, name + '.wav.npy')
            wav_feat = np.load(feats_path)
            # wlength.append(wav_feat.shape[1])
            wav2vec_subfeats.append(wav_feat)

            # trans_path = os.path.join(text_sub_folder, name + '_dys.txt')
            trans_path = os.path.join(text_sub_folder, name + '.txt')
            with open(trans_path, 'r', encoding='utf-8') as f:
                txt = f.readlines()
                if len(txt) == 0:
                    txt = ' '
                transcript = txt[0][:-1]  # get txt information
            # tlength.append(trans_feat.shape[0])
            texts_sub.append(transcript)

            labels_sub.append(label)
    labels.append(labels_sub)
    texts.append(texts_sub)
    wav2vec_feats.append(wav2vec_subfeats)
    return labels, texts, wav2vec_feats


from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def dev_fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer):
    # label二维，text二维，wav2vec二维
    labels, texts, wav2vec_feats= data_load4(data_list, text_sub_folder, wav2vec_sub_folder,
                                                                     label_dict)

    inputs_id = sentence_tokenize1(texts, tokenizer, add_special_tokens=True)

    inputs_id = pack_sequence([torch.Tensor(arr).to(torch.int64) for arr in inputs_id], enforce_sorted=False)
    inputs_id, lengths4 = pad_packed_sequence(inputs_id)
    inputs_id = inputs_id.transpose(0, 1)
    attention_masks = pack_sequence([torch.Tensor(arr).to(torch.int64) for arr in attention_masks], enforce_sorted=False)
    attention_masks, lengths5 = pad_packed_sequence(attention_masks)
    attention_masks = attention_masks.transpose(0, 1)

    wav2vec_feats, a_mask = pad_input_2D(wav2vec_feats, lenght=550)


    wav2vec_feats = pack_sequence([torch.Tensor(arr) for arr in wav2vec_feats], enforce_sorted=False)
    wav2vec_feats, lengths2 = pad_packed_sequence(wav2vec_feats)
    wav2vec_feats = wav2vec_feats.transpose(0, 1)
    lengths2 = lengths2.unsqueeze(0)
    lengths2 = lengths2.expand(wav2vec_feats.shape[1], wav2vec_feats.shape[0])
    lengths2 = lengths2.transpose(0, 1)


    a_mask = pack_sequence([torch.Tensor(arr) for arr in a_mask], enforce_sorted=False)
    a_mask, lengths1 = pad_packed_sequence(a_mask)
    a_mask = a_mask.transpose(0, 1)



    labels = pack_sequence([torch.Tensor(arr) for arr in labels], enforce_sorted=False)
    labels, lengths4 = pad_packed_sequence(labels)
    labels = labels.transpose(0, 1)

    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, a_mask, labels, lengths2)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader


from collections import Counter


def fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer):
    # label一维，text二维，wav2vec二维
    labels, texts, wav2vec_feats = data_load3(data_list, text_sub_folder, wav2vec_sub_folder,
                                                                      label_dict)


    inputs_id = sentence_tokenize(texts, tokenizer, add_special_tokens=True)
    attention_masks = sentence_mask(inputs_id)
    # texts_feats, t_mask = pad_input(texts_feats, lenght=40)
    wav2vec_feats, a_mask = pad_input(wav2vec_feats, lenght=550)



    inputs_id = torch.Tensor(inputs_id).to(torch.int64)
    attention_masks = torch.Tensor(attention_masks).to(torch.int64)


    wav2vec_feats = torch.stack([torch.from_numpy(arr) for arr in wav2vec_feats], dim=0).to(torch.float32)
    a_mask = torch.stack([torch.from_numpy(arr) for arr in a_mask], dim=0)

    labels = torch.Tensor(labels).to(torch.int64)

    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, a_mask, labels)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader


def test_data_load1(data_list, data_folder, feats_folder):
    texts = []
    texts_sub = []
    wav2vec_feats = []
    wav2vec_subfeats = []

    for item in data_list:
        name = item[:-1]
        name, label = name.split(',')
        name = os.path.basename(name).replace('.txt', '')

        # name = name.replace('Data/Scripts_Continuous\\', '').replace('.wav', '')
        if int(name[-2:].replace('_', '')) == 1:
            # names.append(name)
            if len(texts_sub) != 0:
                texts.append(texts_sub)
                wav2vec_feats.append(wav2vec_subfeats)
                texts_sub = []
                wav2vec_subfeats = []
            if name.startswith('ad_'):
                feats_path = os.path.join(feats_folder, name + '.npy')
            else:
                feats_path = os.path.join(feats_folder, name + '.wav.npy')
            wav_feat = np.load(feats_path)
            # wlength.append(wav_feat.shape[1])
            wav2vec_subfeats.append(wav_feat)

            # trans_path = os.path.join(data_folder, name + '_dys.txt')
            trans_path = os.path.join(data_folder, name + '.txt')
            with open(trans_path, 'r', encoding='utf-8') as f:
                txt = f.readlines()
                if len(txt) == 0:
                    txt = ' '
                transcript = txt[0][:-1]  # get txt information
            # tlength.append(trans_feat.shape[0])
            texts_sub.append(transcript)
        else:
            if name.startswith('ad_'):
                feats_path = os.path.join(feats_folder, name + '.npy')
            else:
                feats_path = os.path.join(feats_folder, name + '.wav.npy')
            wav_feat = np.load(feats_path)
            # wlength.append(wav_feat.shape[1])
            wav2vec_subfeats.append(wav_feat)

            # trans_path = os.path.join(data_folder, name + '_dys.txt')
            trans_path = os.path.join(data_folder, name + '.txt')
            with open(trans_path, 'r', encoding='utf-8') as f:
                txt = f.readlines()
                if len(txt) == 0:
                    txt = ' '
                transcript = txt[0][:-1]  # get txt information
            # tlength.append(trans_feat.shape[0])
            texts_sub.append(transcript)
    texts.append(texts_sub)
    wav2vec_feats.append(wav2vec_subfeats)
    return texts, wav2vec_feats


def test1_fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder,tokenizer):
    texts, wav2vec_feats = test_data_load1(data_list, text_sub_folder,
                                                                          wav2vec_sub_folder)

    inputs_id = sentence_tokenize1(texts, tokenizer, add_special_tokens=True)
    attention_masks = sentence_mask1(inputs_id)
    inputs_id = pack_sequence([torch.Tensor(arr).to(torch.int64) for arr in inputs_id], enforce_sorted=False)
    inputs_id, lengths4 = pad_packed_sequence(inputs_id)
    inputs_id = inputs_id.transpose(0, 1)
    attention_masks = pack_sequence([torch.Tensor(arr).to(torch.int64) for arr in attention_masks],
                                    enforce_sorted=False)
    attention_masks, lengths5 = pad_packed_sequence(attention_masks)
    attention_masks = attention_masks.transpose(0, 1)


    wav2vec_feats, a_mask = pad_input_2D(wav2vec_feats, lenght=550)



    wav2vec_feats1 = pack_sequence([torch.Tensor(arr) for arr in wav2vec_feats], enforce_sorted=False)
    wav2vec_feats, lengths2 = pad_packed_sequence(wav2vec_feats1)
    wav2vec_feats = wav2vec_feats.transpose(0, 1)

    lengths2 = lengths2.unsqueeze(0)
    lengths2 = lengths2.expand((wav2vec_feats.shape[1], wav2vec_feats.shape[0]))
    lengths2 = lengths2.transpose(0, 1)

    a_mask1 = pack_sequence([torch.Tensor(arr) for arr in a_mask], enforce_sorted=False)
    a_mask, lengths1 = pad_packed_sequence(a_mask1)
    a_mask = a_mask.transpose(0, 1)



    data = TensorDataset(inputs_id, wav2vec_feats, attention_masks, a_mask, lengths2)
    dataloader = DataLoader(data, batch_size=batch_size, drop_last=True)
    return dataloader


def test_data_load(data_list, data_folder, feats_folder):
    texts_feats = []
    wav2vec_feats = []
    for item in data_list:
        name = item[:-1]
        name, label = name.split(',')
        name = name.replace('Data/Scripts_Continuous\\', '')
        trans_path = os.path.join(data_folder, name + '.text.npy')
        trans_feat = np.load(trans_path)
        # avg_feat = np.mean(trans_feat, axis=0)
        texts_feats.append(trans_feat)
        feats_path = os.path.join(feats_folder, name + '.wav.npy')
        feat = np.load(feats_path)
        # avg_feat = np.mean(feat, axis=0)
        wav2vec_feats.append(feat)

    return texts_feats, wav2vec_feats


def test_fusion_data_load(data_list, text_sub_folder, wav2vec_sub_folder):
    texts_feats, wav2vec_feats = test_data_load(data_list, text_sub_folder, wav2vec_sub_folder)

    texts_feats, t_mask = pad_input(texts_feats, lenght=256)
    wav2vec_feats, a_mask = pad_input(wav2vec_feats, lenght=499)

    texts_feats = torch.stack([torch.from_numpy(arr) for arr in texts_feats], dim=0).to(torch.float32)
    t_mask = torch.stack([torch.from_numpy(arr) for arr in t_mask], dim=0)
    wav2vec_feats = torch.stack([torch.from_numpy(arr) for arr in wav2vec_feats], dim=0).to(torch.float32)
    a_mask = torch.stack([torch.from_numpy(arr) for arr in a_mask], dim=0)

    data = TensorDataset(texts_feats, wav2vec_feats, t_mask, a_mask)
    dataloader = DataLoader(data, batch_size=batch_size, drop_last=True)
    return dataloader
