# 2023年月11日14时31分42秒
import random
import numpy as np
import torch
import time
# from src.timer import format_time
from torch.nn import CrossEntropyLoss

from timer import *
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertConfig
# from src.criteria_cal import flat_accuracy, calculat_f1
from criteria_cal import *
# from src.data_preprocessing import reset_random_seeds
from data_preprocessing import *
import pdb
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
from conf import *
import itertools

import torch.nn.functional as F
from collections import Counter





#
def fine_tune(model, epochs, train_dataloader, aux1_dataloader, aux2_dataloader, dev_dataloader, device,
              model_saved_path):
    reset_random_seeds(SEED)
    loss_values = []
    loss_fct = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                  eps=1e-8, # args.adam_epsilon  - default is 1e-8.
                                  weight_decay=0.01
                                  )

    total_steps = len(train_dataloader) * epochs# # warm_steps=total_steps*0.1
    # # warm_steps=len(train_dataloader)*epochs*0.2

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # For each epoch...
    dev_best_f1 = 0.0

    conterr = 0
    patience = 5
    early_stop = False

    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        t0 = time.time()
        total_loss = 0
        model.train()
        batch1, batch2, batch3=None,None,None
        for step, batch in enumerate(itertools.zip_longest(train_dataloader, aux1_dataloader,aux2_dataloader)):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # model.zero_grad()
            # batch1= batch
            batch1, batch2,batch3 = batch
            if batch1 is None:
                break
            optimizer.zero_grad()
            loss1, loss2, loss3 = 0., 0., 0.
            # kl_loss1, kl_loss2, kl_loss3 = 0., 0., 0.
            b_text = None
            b_wav2vec = None
            b_t_mask = None
            b_a_mask = None
            b_labels = None

            a1_text = None
            a1_wav2vec = None
            a1_t_mask = None
            a1_a_mask = None
            a1_labels = None

            a2_text = None
            a2_wav2vec = None
            a2_t_mask = None
            a2_a_mask = None
            a2_labels = None
            if batch1 is not None:
                b_text = batch1[0].to(device)
                b_wav2vec = batch1[1].to(device)
                b_t_mask = batch1[2].to(device)
                b_a_mask = batch1[3].to(device)
                b_labels = batch1[4].to(device)
            if batch2 is not None:
                a1_text = batch2[0].to(device)
                a1_wav2vec = batch2[1].to(device)
                a1_t_mask = batch2[2].to(device)
                a1_a_mask = batch2[3].to(device)
                a1_labels = batch2[4].to(device)
            if batch3 is not None:
                a2_text = batch3[0].to(device)
                a2_wav2vec = batch3[1].to(device)
                a2_t_mask = batch3[2].to(device)
                a2_a_mask = batch3[3].to(device)
                a2_labels = batch3[4].to(device)

            logits1, logits2, logits3 = model(b_wav2vec, b_text, b_a_mask, b_t_mask,
                                              a1_wav2vec, a1_text, a1_a_mask, a1_t_mask,
                                              a2_wav2vec, a2_text, a2_a_mask, a2_t_mask)

            if b_labels is not None:
                # ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))
                # loss1 = 0.5*loss_fct(logits1.view(-1, 2), b_labels.view(-1))+loss_fct(logits11.view(-1, 2), b_labels.view(-1))
                # kl_loss1 = compute_kl_loss(logits1, logits11)
                loss1 = loss_fct(logits1.view(-1, 2), b_labels.view(-1))
            if a1_labels is not None:
                # loss2 = 0.5*loss_fct(logits2.view(-1, 4), a1_labels.view(-1))+loss_fct(logits22.view(-1, 4), a1_labels.view(-1))
                # kl_loss2 = compute_kl_loss(logits2, logits22)
                loss2 = loss_fct(logits2.view(-1, 4), a1_labels.view(-1))
            if a2_labels is not None:
                # loss3 = 0.5*loss_fct(logits3.view(-1, 2), a2_labels.view(-1))+loss_fct(logits33.view(-1, 2), a2_labels.view(-1))
                # kl_loss3 = compute_kl_loss(logits3, logits33)
                loss3 = loss_fct(logits3.view(-1, 2), a2_labels.view(-1))
            # loss = (loss1 + 5 * kl_loss1)+0.5*(loss2+5*kl_loss2)+(loss3+5*kl_loss3)
            loss = loss1 + 0.5 * loss2 + loss3
            # loss=loss1+loss2/(loss2/loss1)+loss3/(loss3/loss1)
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()

        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy, eval_f_score = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_pred, y_true = [], []
        # Evaluate data for one epoch
        print("***** Running evaluation *****")
        print("  Num examples = {}".format(len(dev_dataloader) * batch_size))
        print("  Epoch = {}".format(epoch_i))

        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            predictions = []
            label_idss = []
            total = []
            b_text, b_wav2vec, t_mask, a_mask, b_labels, length = batch
            for bat in np.arange(b_text.shape[0]):

                b_text1 = b_text[bat][0:length[bat][bat], :]
                b_wav2vec1 = b_wav2vec[bat][0:length[bat][bat], :, :]
                t_mask1 = t_mask[bat][0:length[bat][bat], :].bool()
                a_mask1 = a_mask[bat][0:length[bat][bat], :].bool()
                b_label = b_labels[bat][0:length[bat][bat]]


                with torch.no_grad():
                    outputs = model(b_wav2vec1,
                                    b_text1,
                                    a_mask1,
                                    t_mask1)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()  # (16,2)
                predictions.append(logits)
                # (4,16,2)
                label_id = b_label.to('cpu').numpy()
                label_idss.append(label_id)
                # 4,16
            true_label = []
            for bat in np.arange(b_text.shape[0]):
                true_label.append(int(label_idss[bat][0]))
                tatol = np.argmax(predictions[bat], axis=1)  # (16,)
                counter = Counter(tatol)
                count_0 = counter[0]
                count_1 = counter[1]
                if count_1 > count_0:
                    total.append(1)
                else:
                    total.append(0)
                # 最终total变为（4，），true_label变为（4，）

            tmp_eval_accuracy = flat_accuracy1(np.array(true_label), np.array(total))
            eval_accuracy += tmp_eval_accuracy
            # tmp_eval_f_score = f1_score(true_label, total, average='weighted')
            # eval_f_score += tmp_eval_f_score
            # nb_eval_steps += 1

            pred_flat = np.array(total).flatten()
            y_pred += pred_flat.tolist()
            labels_flat = np.array(true_label).flatten()
            y_true += labels_flat.tolist()

        tmp_eval_f_score = f1_score(y_true, y_pred, average='weighted')


        # if dev_best_f1 < eval_f_score / nb_eval_steps:
        #     torch.save(model.state_dict(), model_saved_path)
        #     dev_best_f1 = eval_f_score / nb_eval_steps
        #     print("best model updated, F score: {0:.2f}".format(eval_f_score / nb_eval_steps))
        if dev_best_f1 < tmp_eval_f_score:
            torch.save(model.state_dict(), model_saved_path)
            dev_best_f1 = tmp_eval_f_score
            print("best model updated, F score: {0:.2f}".format(tmp_eval_f_score))
        else:
            conterr += 1
            if conterr >= patience:
                early_stop= True

        print("  Accuracy: {:.2f}\t"
              "  precision: {:.2f}\t"
              "  f1 score: {:.2f}\t"
              "  confusion matrix: {}".format(accuracy_score(y_true, y_pred),
                                              precision_score(y_true, y_pred, average='weighted'),
                                              f1_score(y_true, y_pred, average='weighted'),
                                              confusion_matrix(y_true, y_pred).tolist()))

        if early_stop:
            torch.save(model.state_dict(), model_saved_path)
            print('Early stopping')
            break

    print("Training complete!")
