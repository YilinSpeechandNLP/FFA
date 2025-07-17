# 2023年月18日20时10分07秒
import json

from click import OptionParser

from GPU_setup import device_setup
from data_preprocessing import *
from conf import *
from BERT_train import fine_tune
from BERT_predict import dev_model_predict, test_model_predict
# from fusion_model import BertForFusion
from result_saving import result_write
from criteria_cal import *
from sklearn.metrics import precision_recall_fscore_support
from model.FFA import build_FFA


# from DWF import model

def main():
    ## set up
    parser = OptionParser()
    parser.add_option("--Fold")
    # parser.add_option("--train_task")
    parser.add_option("--pre_trained_model")
    parser.add_option("--seed")
    options, args = parser.parse_args()
    print(type(options.seed))
    # options={'Fold': '0', 'train_task': 'ADReSSo_test/ADReSSo', 'pre_trained_model': 'bert-base-uncased', 'layer_idx': '8'}
    model_save_folder, results_save_folder, data_list_folder,tokenizer = make_path(options)
    device = device_setup()
    reset_random_seeds(int(options.seed))
    ## data load
    label_dict = np.load(label_dict_path, allow_pickle=True).item()



    print("  Fold = {}".format(options.Fold))
    print("  pre_trained_model = {}".format(options.pre_trained_model))
    Fold=options.Fold
    print(options.Fold)
    Fold = int(Fold) + 1
    train_lst_path = os.path.join(data_list_folder, 'train' + str(Fold) + '_' + str(options.Fold) + '.csv')
    train_lists = open(train_lst_path).readlines()

    dev_lst_path = os.path.join(data_list_folder, 'val' + str(Fold) + '_' + str(options.Fold) + '.csv')
    dev_lists = open(dev_lst_path).readlines()

    test_lst_path = os.path.join(data_list_folder, 'test1' + '.csv')
    test_lists = open(test_lst_path).readlines()

    # wav2vec_feats_folder='features/wav2vec_feats'
    wav2vec_sub_folder = os.path.join(wav2vec_feats_folder)
    # text_sub_folder = os.path.join(text_feats_folder, 'fold_'+options.Fold)
    # text_sub_folder = os.path.join(text_feats_folder, 'fold_' + str('{:04d}'.format(0)))
    text_sub_folder =text_scritps_folder
    print('*****data load start******')
    train_dataloader = fusion_data_load(train_lists, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer)

    aux1_dataloader = aux1_fusion_data_load(train_lists, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer)
    aux2_dataloader = aux2_fusion_data_load(train_lists, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer)

    dev_dataloader = dev_fusion_data_load(dev_lists, text_sub_folder, wav2vec_sub_folder, label_dict,tokenizer)
    test_dataloader = test1_fusion_data_load(test_lists, text_sub_folder, wav2vec_sub_folder,tokenizer)
    print('*****data load finished******')
    model_saved_path = os.path.join(model_save_folder, str(options.seed), 'best_model.pt')

    if os.path.exists(os.path.join(model_save_folder, str(options.seed))) == False:
        os.makedirs(os.path.join(model_save_folder, str(options.seed)))

    ## load pre-trained model
    # model = BertForFusion.from_pretrained(options.pre_trained_model)
    # model = BertForFusion.from_pretrained('./bert-base-uncased')
    # model = BertForFusion()
    with open('./config/model_config.json', 'r') as f1:
        model_json = json.load(f1)['FFA']
    model = build_FFA(**model_json)
    model = model.to(device)

    # model.cuda()
    # model.train()
    ## fine tune the pre_trained model
    # if classification_task == 'BERT_tunning':
    #
    fine_tune(model, epochs, train_dataloader, aux1_dataloader=aux1_dataloader, aux2_dataloader=aux2_dataloader,
              dev_dataloader=dev_dataloader, device=device, model_saved_path=model_saved_path)
    ## get the predict result
    predictions, predict = test_model_predict(model, test_dataloader, device, model_saved_path)
    # predictions = type_check(predictions)
    # pred_test_labels = np.argmax(predictions, axis=1).flatten()

    ## save predicted test label
    # print(predictions.shape)

    label_save_path = os.path.join(results_save_folder, 'pred_labels', str(options.seed),str(options.Fold),
                                   options.pre_trained_model + '_pred_labels')
    if os.path.exists(os.path.join(results_save_folder, 'pred_labels', str(options.seed),str(options.Fold))) == False:
        os.makedirs(os.path.join(results_save_folder, 'pred_labels', str(options.seed),str(options.Fold)))
    np.save(label_save_path, np.asarray(predictions))

    # true_labels, predictions = dev_model_predict(model, dev_dataloader, device, model_saved_path)
    # val_accuracy = flat_accuracy(true_labels, predictions)
    # true_labels = type_check(true_labels)
    # predictions = type_check(predictions)
    # pred_labels = np.argmax(predictions, axis=1).flatten()
    # val_precision, val_recall, val_f_score, _ = precision_recall_fscore_support(true_labels, pred_labels)
    # print("dev set", val_precision, val_recall, val_f_score, val_accuracy)


'''
    ## save dev result
    results_save_path = os.path.join(results_save_folder, options.pre_trained_model+'_val.csv')
    if str(options.Fold) == '0':
        with open(results_save_path,"a+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_data:', test_data_folder, 'train_data:', train_data_folder])
            writer.writerow(["precision_cn","precision_ad","recall_cn","recall_ad","f_score_cn","f_score_ad","accuracy"])
    with open(results_save_path,"a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([val_precision[0], val_precision[1],val_recall[0],val_recall[1], val_f_score[0], val_f_score[1], val_accuracy])
        ## save hidden layer output
'''
if __name__ == "__main__":
    main()
