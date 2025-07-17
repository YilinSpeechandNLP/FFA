# 2023年月13日08时59分06秒
import os

import librosa
import soundfile as sf

def split_audio_by_time(input_file, output_prefix, start_times, end_times):
    print(input_file)
    # 加载语音文件
    audio, sr = librosa.load(input_file, sr=None)
    filename= os.path.basename(input_file)
    # 将时间段转换为样本点
    start_samples = librosa.time_to_samples(start_times, sr=sr)
    end_samples = librosa.time_to_samples(end_times, sr=sr)

    # 分割语音
    for i in range(len(start_samples)):
        # if end_samples[i] - start_samples[i] < 160000:
        #     continue
        segment = audio[start_samples[i]:end_samples[i]]
        output_file = f"{filename.replace('.wav','')}_{i+1}.wav"
        output_file = os.path.join(output_prefix,output_file)
        sf.write(output_file, segment, sr)

test_path='../ADReSSo/audio/test-dist'
test_segmentation='../ADReSSo/audio/test_segmentation'
output_path='../ADReSSo/audio/test-dist-segment'
b=os.listdir(test_segmentation)
# for file in b:
#     start_times=[]
#     end_times=[]
#     with open(os.path.join(test_segmentation,file),'r') as f:
#         lines=f.readlines()
#         for line in lines:
#             line=line.strip().split()
#             line=line[0]
#             lst = line.split(',')
#             if lst[1].strip('"')=='PAR':
#                 start_time=int(lst[2].strip('"'))/1000
#                 end_time=int(lst[3].strip('"'))/1000
#                 if float(end_time) - float(start_time) < 1:
#                     continue
#                 start_times.append(start_time)
#                 end_times.append(end_time)
#         file=os.path.join(test_path,file.split('.')[0]+'.wav')
#         split_audio_by_time(file,output_path,start_times,end_times)

train_path='../ADReSSo/audio/train/'
train_segmentation='../ADReSSo/audio/train_segmentation/ad'
output_path_train='../ADReSSo/audio/train-segment/ad'

#
a=os.listdir(train_segmentation)
for file in a:

    start_times=[]
    end_times=[]
    if os.path.basename(train_segmentation)=='ad':

        with open(os.path.join(train_segmentation,file),'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.strip().split()
                line=line[0]
                lst = line.split(',')
                if lst[1].strip('"')=='PAR':
                    start_time=int(lst[2].strip('"'))/1000
                    # new_row = [float(cell) if 'E' not in cell else "{:.0f}".format(float(cell)) for cell in row]

                    end_time=int(lst[3].strip('"'))/1000 if 'e' not in lst[3].strip('"') else "{:.0f}".format(float(lst[3].strip('"'))/1000)
                    if float(end_time)-float(start_time)<1:
                        continue
                    start_times.append(float(start_time))
                    end_times.append(float(end_time))
            audio_file=os.path.join(train_path,'ad_'+file.split('.')[0]+'.wav')
            split_audio_by_time(audio_file,output_path_train,start_times,end_times)

# # 示例用法
# # input_file = "../ADReSS/ADReSS_test/audio/S160.wav"  # 输入语音文件
# input_file='../ADReSSo/audio/train/ad_adrso024.wav'
# output_prefix = "output_segment"  # 输出语音文件名前缀
# start_times = [0.0, 5.0, 10.0]  # 开始时间段（以秒为单位）
# end_times = [8.778, 8.0001, 15.0]  # 结束时间段（以秒为单位）
#
# split_audio_by_time(input_file, output_prefix, start_times, end_times)