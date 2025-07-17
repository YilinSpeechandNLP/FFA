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

Citation

@article{shao2025modality,
  title={Modality fusion using auxiliary tasks for dementia detection},
  author={Shao, Hangshou and Pan, Yilin and Wang, Yue and Zhang, Yijia},
  journal={Computer Speech \& Language},
  pages={101814},
  year={2025},
  publisher={Elsevier}
}
