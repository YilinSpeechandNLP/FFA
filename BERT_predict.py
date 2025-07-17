# 2023年月11日20时21分11秒
from collections import Counter

import torch
from conf import *
import pdb
import numpy as np


def test_model_predict(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    total = []
    Predict = []
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        predictions = []
        batch = tuple(t.to(device) for t in batch)
        b_text, b_wav2vec, b_t_mask, b_a_mask, length = batch
        for bat in np.arange(b_text.shape[0]):
            b_text1 = b_text[bat][0:length[bat][bat], :]
            b_wav2vec1 = b_wav2vec[bat][0:length[bat][bat], :, :]
            b_t_mask1 = b_t_mask[bat][0:length[bat][bat], :].bool()
            b_a_mask1 = b_a_mask[bat][0:length[bat][bat], :].bool()
            with torch.no_grad():
                outputs = model(b_wav2vec1, b_text1,
                                b_a_mask1,
                                b_t_mask1)
            logits = outputs[0]  # (16,2)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # Store predictions and true labels
            predictions.append(logits)  # (4,16,2)
        for bat in np.arange(b_text.shape[0]):
            tatol = np.argmax(predictions[bat], axis=1)  # (16,)
            Predict.append(predictions)
            counter = Counter(tatol)
            # if len(predictions[bat]) <7:
            #     b=2
            # else:
            #     b=len(predictions[bat]) //3
            if counter[1] >= counter[0]:
                total.append(1)
            else:
                total.append(0)

    print('DONE.')
    return total, Predict


def dev_model_predict(model, test_dataloader, device, best_model_path):
    # Put model in evaluation mode
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        b_text, b_wav2vec, b_t_mask, b_a_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_wav2vec, b_text,
                            b_a_mask, b_t_mask
                            )
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')
    return true_labels, predictions
