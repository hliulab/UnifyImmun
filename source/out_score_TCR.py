import warnings
from models.TCR import *
import torch
from tqdm import tqdm
import argparse
import pandas as pd
import torch.utils.data as Data
warnings.filterwarnings("ignore")
model_pred ,data_label= [],[]
num_workers = 1
# vocab_dict = np.load('./data/data_dict.npy',allow_pickle=True).item()
# The parameters of input
argparser = argparse.ArgumentParser()
argparser.add_argument('--input', type=str, help='the path to the input data file (*.csv)',required=True)
argparser.add_argument('--output', type=str, help='the path to the output data file (*.csv)', required=True)
args = argparser.parse_args()

def make_data(data):
    pep_inputs, tcr_inputs = [], []
    for pep, tcr in zip(data.peptide, data.tcr):
        pep, tcr = pep.ljust(tcr_max_len, '-'), tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
    return torch.LongTensor(pep_inputs), torch.LongTensor(tcr_inputs)
class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx]

def data_with_loader(batch_size=batch_size):
    data = pd.read_csv('{}.csv'.format(args.input))
    pep = data[['peptide']]
    tcr = data[['tcr']]
    pep_inputs, tcr_inputs = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, tcr_inputs), batch_size, shuffle=False, num_workers=0, drop_last=True)
    return loader,pep,tcr

loader,pep,tcr = data_with_loader(
    batch_size=batch_size)

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
    return model_pred

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model = Mymodel_tcr().to(device)
    model_path = './trained_model/TCR_2/model_TCR.pkl'
    model.load_state_dict(torch.load(model_path,map_location=device))
    model_eval = model.eval()
    pred = eval_step(model_eval, loader)
    pep = pep.head(len(pred))
    hla = tcr.head(len(pred))
    df = pd.DataFrame({
        'peptide': pep['peptide'].values,
        'tcr': hla['tcr'].values,
        'pred': model_pred
    })

    df.to_csv('{}.csv'.format(args.output), index=False)