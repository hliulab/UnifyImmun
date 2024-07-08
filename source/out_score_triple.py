import warnings
from models.HLA import *
from models.TCR import *
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import torch.utils.data as Data
warnings.filterwarnings("ignore")
model_pred ,data_label= [],[]
num_workers = 1
# vocab_dict = np.load('../data/data_dict.npy',allow_pickle=True).item()


# The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--input', type=str, help='the path to the input data file (*.csv)',required=True)
argparser.add_argument('--output', type=str, help='the path to the output data file (*.csv)', required=True)
args = argparser.parse_args()

def make_data(data):
    pep_inputs, hla_inputs, tcr_inputs = [], [], []
    for tcr, pep, hla in zip(data.tcr, data.peptide, data.HLA):
        tcr, pep, hla = tcr.ljust(hla_max_len, '-'), pep.ljust(hla_max_len, '-'), hla.ljust(hla_max_len, '-')
        tcr_input = [[vocab[n] for n in tcr]]
        pep_input = [[vocab[n] for n in pep]]
        hla_input = [[vocab[n] for n in hla]]
        tcr_inputs.extend(tcr_input)
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
    return torch.LongTensor(tcr_inputs), torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs)
class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx]

def data_with_loader(batch_size=batch_size):
    data = pd.read_csv('{}.csv'.format(args.input))
    tcr = data[['tcr']]
    pep = data[['peptide']]
    hla = data[['HLA']]
    tcr_inputs, pep_inputs, hla_inputs = make_data(data)
    loader_hla = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs), batch_size, shuffle=False, num_workers=0, drop_last=True)
    loader_tcr = Data.DataLoader(MyDataSet(pep_inputs, tcr_inputs), batch_size, shuffle=False, num_workers=0, drop_last=True)
    return loader_hla,loader_tcr,tcr,pep,hla

loader_hla,loader_tcr,tcr,pep,hla = data_with_loader(batch_size=batch_size)

def eval_step(model, val_loader):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs in tqdm(val_loader,colour='cyan'):
            val_pep_inputs, val_hla_inputs = val_pep_inputs.to(device), val_hla_inputs.to(device)
            val_outputs,val_dec_self_attns = model(val_pep_inputs, val_hla_inputs)
            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            model_pred.extend(y_prob_val)
            y_prob_val_list.extend(y_prob_val)
    return y_prob_val_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model_HLA = Mymodel_HLA().to(device)
    model_path_HLA = './trained_model/HLA_2/model_HLA.pkl'
    model_HLA.load_state_dict(torch.load(model_path_HLA,map_location=device))
    model_eval_HLA = model_HLA.eval()
    model_TCR = Mymodel_tcr().to(device)
    model_path_TCR = './trained_model/TCR_2/model_TCR.pkl'
    model_TCR.load_state_dict(torch.load(model_path_TCR,map_location=device))
    model_eval_TCR = model_TCR.eval()
    pred_HLA = eval_step(model_eval_HLA, loader_hla)
    pred_TCR = eval_step(model_eval_TCR, loader_tcr)
    tcr = tcr.head(len(pred_TCR))
    pep = pep.head(len(pred_HLA))
    hla = hla.head(len(pred_HLA))
    df = pd.DataFrame({
        'tcr':tcr['tcr'].values,
        'peptide': pep['peptide'].values,
        'HLA': hla['HLA'].values,
        'HLA_pred': pred_HLA,
        'TCR_pred': pred_TCR
    })

    df.to_csv('{}.csv'.format(args.output), index=False)
