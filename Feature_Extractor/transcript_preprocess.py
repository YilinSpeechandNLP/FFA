# 2023年月14日16时01分44秒
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import numpy as np
import torch
import librosa

tokenizer = Wav2Vec2Processor.from_pretrained("../facebook/wav2vec2-large-960h-lv60")
model = Wav2Vec2ForCTC.from_pretrained("../facebook/wav2vec2-large-960h-lv60")

def wav_to_transcript(wav_path):
    audio_input, fs = librosa.load(wav_path, sr=16000)
    if len(audio_input.shape) > 1:
        audio_input = audio_input[:, 0] + audio_input[:, 1]
    input_values = tokenizer(audio_input, sampling_rate=16000,return_tensors="pt").input_values  # Batch size 1
    outputs = model(input_values, output_hidden_states=True)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]
    return transcription
def sort_by_number(file_name):
    # 提取文件名中的数字部分
    number = int(''.join(filter(str.isdigit, file_name)))
    return number

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
    output_path='feature/scritps'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    wav_path='../ADReSSo/audio/test-dist-segment'
    wav_list=os.listdir(wav_path)
    wav_list=sorted(wav_list, key=sort_by_number)
    for i in wav_list:
        wav=os.path.join(wav_path,i)
        transcript=wav_to_transcript(wav).lower()

        with open(os.path.join(output_path,i.replace('.wav','.txt')),'w') as f:
            f.write(transcript)
        print(transcript)