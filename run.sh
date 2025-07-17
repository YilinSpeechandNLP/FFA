#! /bin/bash



#for layer_idx in {0..24}; do
#for pre_trained_model in roberta-base;do
for pre_trained_model in bert-base-uncased;do
  for sed in 123 ;do
    for idx in {0..9};do
      echo $idx
      echo ${sed}
  #    echo ${tlength[idx]}
  #    echo ${wlength[idx]}
      python BERT_fusion.py --Fold=$idx --pre_trained_model=$pre_trained_model --seed=${sed}

    done
  done
done
#done
