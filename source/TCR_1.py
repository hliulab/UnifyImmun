import time
from models.TCR import *
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

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup1 = {}
        self.backup2 = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == 'encoder_T.src_emb':
                    self.backup1[name] = param.data.clone()
                if emb_name == 'encoder_P.src_emb':
                    self.backup2[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:

                if emb_name == 'encoder_T.src_emb':
                    assert name in self.backup1
                    param.data = self.backup1[name]
                if emb_name == 'encoder_P.src_emb':
                    assert name in self.backup2
                    param.data = self.backup2[name]
        if emb_name == 'encoder_T.src_emb':
            self.backup1 = {}
        if emb_name == 'encoder_P.src_emb':
            self.backup2 = {}


def train_tcr(model, train_loader, fold, epoch, epochs):
    model.encoder_P.load_state_dict(torch.load('../trained_model/HLA_1/encoder_P_{}.pth'.format(fold)))
    train_time = 0
    model.train()
    y_true_list, y_pred_list,attention_list = [], [],[]
    loss_list = []
    fgm = FGM(model)
    for train_pep_inputs, train_tcr_inputs, train_labels in tqdm(train_loader,colour='yellow'):
        train_pep_inputs, train_tcr_inputs, train_labels = train_pep_inputs.to(device), train_tcr_inputs.to(device), train_labels.to(device)
        start = time.time()
        train_outputs,cross_attention = model(train_pep_inputs, train_tcr_inputs)
        train_loss = criterion_tcr(train_outputs, train_labels)
        train_loss.backward()
        train_time += time.time() - start
        fgm.attack(emb_name='encoder_T.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        train_outputs2, train_dec_self_attns2 = model(train_pep_inputs, train_tcr_inputs)
        loss_sum = criterion_tcr(train_outputs2, train_labels)
        loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
        fgm.restore(emb_name='encoder_T.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        optimizer_tcr.step()       #更新模型参数
        optimizer_tcr.zero_grad()  #将所有参数的梯度清零
        y_true_train = train_labels.cpu().numpy()
        y_pred_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_list.extend(y_true_train)
        y_pred_list.extend(y_pred_train)
        loss_list.append(train_loss)
        attention_list.append(cross_attention)
    y_pred_transfer_train_list = transfer(y_pred_list, threshold)
    result_train = (y_true_list, y_pred_list, y_pred_transfer_train_list)
    print('Fold-{} Train: Epoch:{}/{} Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs,f_mean(loss_list),train_time))
    performance_train = performance(y_true_list, y_pred_list, y_pred_transfer_train_list)
    return result_train, performance_train, train_time, attention_list

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
for fold in range(1, 6):
    print('Fold-{}:'.format(fold))
    print('Load TCR Data:')
    train_loader_tcr = data_load_tcr(type_='train', fold=fold, batch_size=batch_size)
    val_loader_tcr = data_load_tcr(type_='val', fold=fold, batch_size=batch_size)
    train_data_tcr = pd.read_csv('../data/data_TCR/train_fold_{}.csv'.format(fold)).dropna()
    val_data_tcr = pd.read_csv('../data/data_TCR/val_fold_{}.csv'.format(fold)).dropna()
    print('Fold-{} Label: Train = {} | Val = {}'.format(fold, Counter(train_data_tcr.label),Counter(val_data_tcr.label)))
    print('TCR Train:')
    path_all_tcr = '../trained_model/TCR_1'
    save_path_tcr = '../trained_model/TCR_1/model_TCR_fold{}.pkl'.format(fold)
    encoder_path_tcr = '../trained_model/TCR_1/encoder_P_{}.pth'.format(fold)
    print('save path: ', save_path_tcr)
    performance_best_tcr, epoch_best_tcr = 0, -1
    time_train = 0
    for epoch in range(1, epochs + 1):
        result_train_tcr, performance_train_tcr, time_train_ep_tcr, attention_score_tcr = train_tcr(model_tcr,train_loader_tcr,fold, epoch,epochs)
        result_val_tcr, performance_val_tcr, attention_score_val_tcr = valid_tcr(model_tcr, val_loader_tcr, fold,epoch, epochs)
        performance_avg = sum(performance_val_tcr[:5]) / 5
        if performance_avg > performance_best_tcr:
            performance_best_tcr, epoch_best_tcr = performance_avg, epoch
            if not os.path.exists(path_all_tcr):
                os.makedirs(path_all_tcr)
            print('****Saving model: Best epoch = {} | metrics_Best_avg = {:.4f}'.format(epoch_best_tcr,
                                                                                         performance_best_tcr))
            print('*****Path saver: ', save_path_tcr)
            torch.save(model_tcr.eval().state_dict(), save_path_tcr)
            torch.save(model_tcr.eval().encoder_P.state_dict(), encoder_path_tcr)
        time_train += time_train_ep_tcr
    print('TCR Training Finished')
    print('-----Evaluate Results-----')
    if epoch_best_tcr >= 0:
        print('*****Path saver: ', save_path_tcr)
        model_tcr.load_state_dict(torch.load(save_path_tcr))
        model_eval_tcr = model_tcr.eval()
        independent_result_tcr, independent_performance_tcr, independent_attention_tcr = valid_tcr(model_eval_tcr, independent_loader_tcr,fold, epoch_best_tcr, epochs)
        covid_result_tcr, covid_performance_tcr, covid_attention_tcr = valid_tcr(model_eval_tcr, covid_loader_tcr,fold, epoch_best_tcr, epochs)
        tripple_result_tcr, tripple_performance_tcr, tripple_attention_tcr = valid_tcr(model_eval_tcr,tripple_loader_tcr, fold, epoch_best_tcr, epochs)

        independent_fold_performance_list_tcr.append(independent_performance_tcr)
        covid_fold_performance_list_tcr.append(covid_performance_tcr)
        tripple_fold_performance_list_tcr.append(tripple_performance_tcr)

end_time = time.time()
use_time = end_time-start_time
print('Use Time:{:6.2f}seconds'.format(use_time))
print('****new_data_TCR_10x Independent set:')
print(performance_pd(independent_fold_performance_list_tcr).to_string())
print('****new_data_TCR_10x Covid set:')
print(performance_pd(covid_fold_performance_list_tcr).to_string())
print('****new_data_TCR_10x Triple set:')
print(performance_pd(tripple_fold_performance_list_tcr).to_string())
