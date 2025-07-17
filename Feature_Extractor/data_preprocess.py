import os
import scipy.signal as signal

import soundfile


import librosa
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
# import fairseq
# import lmdb
import shutil
import math
from WavLM import WavLM,WavLMConfig

'''
目标:将一整段很长的语音进行切割，只保留
'''

import torch
from WavLM import WavLM, WavLMConfig


def extract_wav2vec(wavfile,model):
    print('Extracting wav2vec feature...')
    print(wavfile)


    data, _ = librosa.load(wavfile, sr=16000)#1421797
    data = data[np.newaxis, :]#1,1421797
    data = torch.Tensor(data).to(device)
    if cfg.normalize:
        data = torch.nn.functional.layer_norm(data, data.shape)
    with torch.no_grad():
        feature = model.extract_features(data)[0]#1,4442,1024

    return feature
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False

    checkpoint = torch.load('pre/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ADReSSo_wavpath='../ADReSSo/audio/train-segment/ad'
    # b=os.listdir(ADReSSo_wavpath)
    # ADReSS_out_save_path='feature/ADReSSo/'
    # for i in b:
    #     wavfile = ADReSSo_wavpath + '/' + i
    #     feature = extract_wav2vec(wavfile,model)
    #     i=i.replace('.wav','')
    #     np.save(ADReSS_out_save_path + i + '.npy', feature)
    #
    # ADReSSo_wavpath2='../ADReSSo/audio/train-segment/cn'
    # b_test=os.listdir(ADReSSo_wavpath2)
    # for i in b_test:
    #     wavfile=ADReSSo_wavpath2+'/'+i
    #     feature=extract_wav2vec(wavfile,model)
    #     np.save(ADReSS_out_save_path+i+'.npy',feature)


    ADReSSo_wavpath3= '../ADReSSo/audio/test-dist-segment'
    a=os.listdir(ADReSSo_wavpath3)
    ADReSSo_out_save_path='feature/ADReSSo/'
    for i in a:
        wavfile=ADReSSo_wavpath3+'/'+i
        feature=extract_wav2vec(wavfile,model)
        np.save(ADReSSo_out_save_path+i+'.npy',feature)

    # ADReSSo_wavpath2='../ADReSSo/audio/test-dist'
    # a_test=os.listdir(ADReSSo_wavpath2)
    # for i in a_test:
    #     wavfile=ADReSSo_wavpath2+'/'+i
    #     feature=extract_wav2vec(wavfile,model)
    #     np.save(ADReSSo_out_save_path+i+'.npy',feature)






