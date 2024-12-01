import time
from models.TCR import *
#from scipy import interp
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
tcr_max_len = 34
vocab_size = len(vocab)
n_heads = 1
start_time = time.time()
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1
threshold = 0.5
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
model_tcr = Mymodel_tcr().to(device)
criterion_tcr = nn.CrossEntropyLoss()
optimizer_tcr = optim.Adam(model_tcr.parameters(), lr=1e-3)


def performance(y_true, y_pred,y_pred_transfer):
    accuracy = accuracy_score(y_true=y_true,y_pred=y_pred_transfer)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
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




def performance_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']

    performance_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performance_pd.loc['mean'] = performance_pd.mean(axis=0)
    performance_pd.loc['std'] = performance_pd.std(axis=0)

    return performance_pd

def valid_tcr(model, val_loader, fold, epoch, epochs):

    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        y_true_val_list, y_pred_val_list, attention_val_list = [], [], []
        loss_val_list = []
        for val_pep_inputs, val_tcr_inputs, val_labels in tqdm(val_loader,colour='blue'):
            val_pep_inputs, val_tcr_inputs, val_labels = val_pep_inputs.to(device), val_tcr_inputs.to(device), val_labels.to(device)
            val_outputs,cross_attention_val = model(val_pep_inputs, val_tcr_inputs)
            val_loss = criterion_tcr(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            # y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            loss_val_list.append(val_loss)
        y_pred_transfer_val_list = transfer(y_pred_val_list, threshold)
        result_val = (y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)

        print('Fold-{} Valid: Epoch:{}/{} Loss = {:.4f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        performance_val = performance(y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)
    return result_val, performance_val, cross_attention_val

independent_loader_tcr = data_load_tcr(type_='independent', fold=None, batch_size=batch_size)
covid_loader_tcr= data_load_tcr(type_='covid', fold=None, batch_size=batch_size)
tripple_loader_tcr = data_load_tcr(type_='triple', fold=None, batch_size=batch_size)





train_fold_performance_list_tcr, val_fold_performance_list_tcr, independent_fold_performance_list_tcr, covid_fold_performance_list_tcr, tripple_fold_performance_list_tcr = [], [], [], [], []

attention_train_dict, attention_val_dict, attention_independent_dict, attention_external_dict = {}, {}, {}, {}
for fold in range(4, 5):
    print('Fold-{}:'.format(fold))
    save_path_tcr = '../trained_model/TCR_2/model_TCR.pkl'
    print('save path: ', save_path_tcr)

    print('-----Evaluate Results-----')
    print('*****Path saver: ', save_path_tcr)
    model_tcr.load_state_dict(torch.load(save_path_tcr,device))
    model_eval_tcr = model_tcr.eval()
    independent_result_tcr, independent_performance_tcr, independent_attention_tcr = valid_tcr(model_eval_tcr, independent_loader_tcr,fold, 0, epochs)
    covid_result_tcr, covid_performance_tcr, covid_attention_tcr = valid_tcr(model_eval_tcr, covid_loader_tcr,fold, 0, epochs)
    tripple_result_tcr, tripple_performance_tcr, tripple_attention_tcr = valid_tcr(model_eval_tcr,tripple_loader_tcr, fold, 0, epochs)

    independent_fold_performance_list_tcr.append(independent_performance_tcr)
    covid_fold_performance_list_tcr.append(covid_performance_tcr)
    tripple_fold_performance_list_tcr.append(tripple_performance_tcr)

end_time = time.time()
use_time = end_time-start_time
print('Use Time:{:6.2f}seconds'.format(use_time))
print('****data_TCR Independent set:')
print(performance_pd(independent_fold_performance_list_tcr).to_string())
print('****data_TCR Covid set:')
print(performance_pd(covid_fold_performance_list_tcr).to_string())
print('****data_TCR Triple set:')
print(performance_pd(tripple_fold_performance_list_tcr).to_string())
