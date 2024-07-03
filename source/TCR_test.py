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
def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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




def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']

    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis=0)
    performances_pd.loc['std'] = performances_pd.std(axis=0)

    return performances_pd


def valid_HLA(model, val_loader, fold, epoch, epochs):
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
external_loader = data_load_tcr(type_='covid', fold=None, batch_size=batch_size)
triple_loader = data_load_tcr(type_='triple', fold=None, batch_size=batch_size)



train_fold_performance_list, val_fold_performance_list,independent_fold_performance_list, external_fold_performance_list = [], [], [], []
triple_fold_metrics_list = []
attention_train_dict, attention_val_dict, attention_independent_dict, attention_external_dict = {}, {}, {}, {}

for fold in range(9, 10):
    save_path = '../trained_model/TCR_2/model_TCR.pkl'
    # encoder_path = 'UnifyImmun/HLA_1/encoder_P_{}.pth'.format(fold)
    print('save path: ', save_path)
    performance_best, epoch_best = 0, -1
    time_train = 0
    model.load_state_dict(torch.load(save_path,map_location=device))
    model_eval = model.eval()
    independent_result, independent_performance, independent_attention = valid_HLA(model_eval,independent_loader, fold,epoch_best, epochs)
    external_result, external_performance, external_attention = valid_HLA(model_eval, external_loader, fold,epoch_best, epochs)
    triple_result, triple_performance, triple_attention = valid_HLA(model_eval, triple_loader, fold,epoch_best, epochs)

    independent_fold_performance_list.append(independent_performance)
    external_fold_performance_list.append(external_performance)
    triple_fold_metrics_list.append(triple_performance)
    print("Total training time: {:6.2f} sec".format(time_train))


print('****Independent set:')
print(performances_to_pd(independent_fold_performance_list).to_string())
print('****Covid set:')
print(performances_to_pd(external_fold_performance_list).to_string())
print('****Triple set:')
print(performances_to_pd(triple_fold_metrics_list).to_string())

