dataset：
首先下载ADReSSo的数据集
	文件目录介绍：
		test-dist、train：整段音频
		test-dist-segment、train-segment：分割后的语音片段
		train_segmentation:音频记录文件

在Feature_Extractor目录下进行语音分割、转录、声学特征提取。
语音分割：audio_segmentation.py
转录：transcript_preprocess.py
声学特征提取：acoustic_feature_extraction.py、acoustic_feature_extraction.py
其中文件中的目录修改为自己文件中路径即可。

运行：
```
sh run.sh
``