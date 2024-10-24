import time
from models.TCR import *
import random
from scipy import interp
import warnings
import inputs as inputs_lib
from collections import Counter
from functools import reduce
#import tensorflow
from tqdm import tqdm, trange
from copy import deepcopy
from sklearn.metrics import confusion_matrix,matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc,accuracy_score,f1_score
from sklearn.metrics import precision_recall_curve,precision_score,recall_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
seed = 66
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
pep_max_len = 15
hla_max_len = 34
tcr_max_len = 34
tgt_len = pep_max_len + hla_max_len
vocab_size = len(vocab)
n_heads = 1
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1
threshold = 0.5
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
model = Mymodel_tcr().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def performance(y_true, y_pred,y_pred_transfer):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    accuracy = accuracy_score(y_true=y_true,y_pred=y_pred_transfer)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true=y_true,y_pred=y_pred_transfer)
    recall = recall_score(y_true=y_true,y_pred=y_pred_transfer)
    f1 = f1_score(y_true=y_true,y_pred=y_pred_transfer)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true,y_pred_transfer)
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred_transfer)[0], Counter(y_pred_transfer)[1]))
    print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity,
                                                                                          specificity, accuracy,mcc
                                                                                          ))
    print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))

    return (roc_auc, accuracy, mcc, f1, aupr,sensitivity, specificity, precision, recall )




f_mean = lambda l: sum(l) / len(l)



def valid_TCR(model, val_loader, fold, epoch, epochs):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        y_true_val_list, y_pred_val_list, attention_val_list = [], [], []
        loss_val_list = []
        for val_anti_inputs, val_hla_inputs, val_labels in tqdm(val_loader,colour='blue'):
            val_anti_inputs, val_hla_inputs, val_labels = val_anti_inputs.to(device), val_hla_inputs.to(device), val_labels.to(device)
            val_outputs,cross_attention_val = model(val_anti_inputs, val_hla_inputs)
            val_loss = criterion(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            loss_val_list.append(val_loss)
        y_pred_transfer_val_list = transfer(y_pred_val_list, threshold)
        result_val = (y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)

        print('Fold-{} Valid: Epoch:{}/{} Loss = {:.4f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        performance_val = performance(y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)
    return result_val, performance_val, cross_attention_val





independent_loader = data_load_tcr(type_='independent', fold=None, batch_size=batch_size)
covid_loader = data_load_tcr(type_='covid', fold=None, batch_size=batch_size)
triple_loader = data_load_tcr(type_='triple', fold=None, batch_size=batch_size)



for fold in [1]:
    save_path = '../trained_model/TCR_2/model_TCR_2x.pkl'
    print('save path: ', save_path)
    performance_best, epoch_best = 0, -1
    time_train = 0
    model.load_state_dict(torch.load(save_path,map_location=device))
    model_eval = model.eval()
    independent_result, independent_performance, independent_attention = valid_TCR(model_eval,independent_loader, fold,epoch_best, epochs)
    covid_result, covid_performance, covid_attention = valid_TCR(model_eval, covid_loader, fold,epoch_best, epochs)
    triple_result, triple_performance, triple_attention = valid_TCR(model_eval, triple_loader, fold,epoch_best, epochs)

    independent_result_data_dict = {}
    covid_result_data_dict = {}
    triple_result_data_dict = {}

    for i in range(len(independent_result[0])):
        y_true = independent_result[0][i].item()
        y_pred = independent_result[1][i].item()
        y_prob = independent_result[2][i].item()

        independent_result_data_dict[i] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    for i in range(len(covid_result[0])):
        y_true = covid_result[0][i].item()
        y_pred = covid_result[1][i].item()
        y_prob = covid_result[2][i].item()

        covid_result_data_dict[i] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    for i in range(len(triple_result[0])):
        y_true = triple_result[0][i].item()
        y_pred = triple_result[1][i].item()
        y_prob = triple_result[2][i].item()


        triple_result_data_dict[i] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    # Convert the result_data_dict values to a list
    independent_result_data = list(independent_result_data_dict.values())
    covid_result_data = list(covid_result_data_dict.values())
    triple_result_data = list(triple_result_data_dict.values())

    independent_result_df = pd.DataFrame(independent_result_data)
    covid_result_df = pd.DataFrame(covid_result_data)
    triple_result_df = pd.DataFrame(triple_result_data)
    # Save the DataFrame to a CSV file
    independent_result_df.to_csv('independent_result_data.csv', index=False)
    covid_result_df.to_csv('covid_result_data.csv', index=False)
    triple_result_df.to_csv('external_result_data.csv', index=False)

