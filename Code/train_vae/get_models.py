import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from datetime import timedelta
import time
import pandas as pd


class Progress_Bar():
    def __init__(self, epochs, steps_per_epoch, metrics, avg_over_n_steps=100, sleep_print=0.25):

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch 
        self.total_steps = self.epochs * steps_per_epoch
        self.remaining_steps = self.epochs * steps_per_epoch
        self.avg_over_n_steps = avg_over_n_steps
        self.tic = time.time() 
        self.sleep_print = sleep_print
        self.iteration = 0
        self.metrics = metrics
        self.time_metrics = ['Progress', 'ETA', 'Epoch', 'Iteration', 'ms/Iteration']
        
        self.dict = {
                    'Progress' : "0.000%",
                    'ETA' : 0.0,
                    'Epoch' : int(1),
                    'Iteration' : int(0),
                    'ms/Iteration' : 0.0,
                    'time': [time.time()]
                    }

        self.dict.update({metric: [] for metric in metrics})
        self.dict.update({metric+' last ep': [] for metric in metrics})
        self.dict.update({metric+' impr': 0.0 for metric in metrics})
        
    @staticmethod
    def format_number(number, min_length):
        decimal_count = len(str(number).split('.')[0])  
        decimal_places = max(min_length - decimal_count, 0) 

        formatted_number = "{:.{}f}".format(number, decimal_places)
        return formatted_number
    
    def ret_sign(self, number, min_length):
        if number > 0.0:
            sign_str = '\033[92m{}\033[00m'.format("+" + self.format_number(np.abs(number), min_length))
        elif number < 0.0:
            sign_str = '\033[91m{}\033[00m'.format("-" + self.format_number(np.abs(number), min_length))
        else:
            sign_str = '---'
        return  sign_str      

    def update(self, values):   
        self.remaining_steps -= 1   
        for key, value in values.items():
            self.dict[key].append(value) 
            
        if self.dict['Iteration'] == 1:
            for key, value in values.items():
                self.dict[key+' last ep'].append(value) 
             
        self.dict['Iteration'] += 1
        
        epoch = int(np.ceil(self.dict['Iteration'] / self.steps_per_epoch))

        if self.dict['Epoch'] < epoch:
            for key in self.metrics:
                self.dict[key+' last ep'].append(np.mean(self.dict[key][-self.steps_per_epoch:]))
                self.dict[key+' impr'] = self.dict[key+' last ep'][-2] - self.dict[key+' last ep'][-1]
            self.dict['Epoch'] = epoch
     
        self.dict['time'].append(time.time())
  
        avg_steps = np.min((self.dict['Iteration'], self.avg_over_n_steps))
        avg_time = (self.dict['time'][-1] - self.dict['time'][-avg_steps-1]) / avg_steps 
        
        self.dict['ETA'] = timedelta(seconds=int(self.remaining_steps * avg_time))         
        self.dict['ms/Iteration'] = self.format_number(avg_time*1000.0, 4)
        self.dict['Progress'] = self.format_number(100.0 * self.dict['Iteration'] / self.total_steps, 3)+'%'

        if time.time() - self.tic > self.sleep_print:
            metric_string =  [f'\033[95m{key}\033[00m: {self.dict[key]}' for key in self.time_metrics]       
            metric_string += [f'\033[33m{key}\033[00m: {self.format_number(np.mean(self.dict[key][-avg_steps:]), 5)} ({self.ret_sign(self.dict[key+" impr"], 4)})' for key in self.metrics]               
            metric_string =  "\033[96m - \033[00m".join(metric_string)
            print(f"\r{metric_string}.           ", end='', flush=True)   
            self.tic = time.time()      
            
            
def create_structure(neurons, 
                     act,
                     use_bn,
                     use_ln,
                     dropout,
                     layer_order=['linear', 'batch_norm', 'layer_norm', 'act', 'dropout']
                     ):
    layers = []

    for neurons_in, neurons_out in zip(neurons, neurons[1:]):
        for operation in layer_order:
            if operation == 'linear':
                layers.append(nn.Linear(neurons_in, neurons_out))
            elif operation == 'act' and act != None:
                layers.append(act)                      
            elif operation == 'dropout' and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            elif operation == 'layer_norm' and use_ln:
                layers.append(nn.LayerNorm(neurons_out))
            elif operation == 'batch_norm' and use_bn:
                layers.append(nn.BatchNorm1d(neurons_out))   
                            
    return nn.Sequential(*layers)

  
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 act,                  
                 hidden_dims,             
                 output_dim,  
                 use_bn=False,
                 use_ln=False,
                 dropout=0.0,
                 additional_layers_dims=False
                 ):
        super(Encoder, self).__init__()

        self.model = create_structure(neurons=[input_dim]+hidden_dims, 
                                      act=act,
                                      use_bn=use_bn,
                                      use_ln=use_ln,
                                      dropout=dropout,
                                      )
        
        # Zusätzliche Schichten
        if additional_layers_dims:
            self.additional_layers = create_structure(neurons=[hidden_dims[-1]]+additional_layers_dims,
                                                      act=act,
                                                      use_bn=use_bn,
                                                      use_ln=use_ln,
                                                      dropout=dropout,
                                                      )
        else:
            self.additional_layers = nn.Identity()  # Falls keine zusätzlichen Schichten definiert sind

        final_hidden_dim = additional_layers_dims[-1] if additional_layers_dims else hidden_dims[-1]
        self.mu = nn.Linear(final_hidden_dim, output_dim)
        self.log_sig = nn.Linear(final_hidden_dim, output_dim)

        self.prior = Normal(
            torch.zeros(torch.Size([output_dim])), 
            torch.ones(torch.Size([output_dim])))

    def encode(self, input):
        x = self.model(input)
        x = self.additional_layers(x)
        mu = self.mu(x)
        log_sig = self.log_sig(x)
        return mu, log_sig

    def sample(self, mu, log_sig):
        eps = self.prior.sample(torch.Size([log_sig.size(dim=0)])) 
        z = mu + log_sig.exp() * eps
        return z

    def calc_kl_div(self, mu, log_sig):
        log_sig_sq = 2.0 * log_sig
        return torch.mean(0.5 * torch.sum(mu.square() + torch.exp(log_sig_sq) - 1.0 - log_sig_sq, dim=1))

    def forward(self, input):
        mu, log_sig = self.encode(input)
        kl_div = self.calc_kl_div(mu, log_sig)
        z = self.sample(mu, log_sig)
        return z, mu, kl_div
    

class Exp_CumSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cumsum(torch.exp(x).clamp(min=1e-8), dim=-1)


class Decoder(nn.Module):
    def __init__(self, 
                 item_positions,
                 act,                  
                 input_dim,  
                 hidden_dims,             
                 use_bn = False,
                 use_ln = False,
                 dropout = 0.0,
                 ):
        super(Decoder, self).__init__()

        self.model = create_structure(neurons=[input_dim]+hidden_dims, 
                                      act=act,
                                      use_bn=use_bn,
                                      use_ln=use_ln,
                                      dropout=dropout,
                                      )

        self.items, num_answers = np.unique(item_positions, return_counts=True)
        self.num_cuts = num_answers
        self.f = nn.Linear(hidden_dims[-1], len(self.items))
        self.k = nn.ModuleList([nn.Sequential(*[nn.Linear(hidden_dims[-1], cp)], Exp_CumSum()) for cp in self.num_cuts])
        self.link_fn = F.sigmoid

       
    def decode(self, input):
        x = self.model(input)
        f_x = self.f(x)
        cutpoints = [layer(x) for layer in self.k]  
        cum_probs = [self.link_fn(cutpoints[i] - f_x[:,i].unsqueeze(1).repeat(1, cp)).clamp(1e-8, 1-1e-8) for i,cp in enumerate(self.num_cuts)]    
        probs = [torch.stack([cum_probs[j][:,0]] + [cum_probs[j][:,i]-cum_probs[j][:,i-1]for i in range(1,cp)] + [1.0-cum_probs[j][:,-1]]).t().clamp(1e-8, 1-1e-8) for j,cp in enumerate(self.num_cuts)]

        return probs

    def calc_rec_loss(self, probs, data):
        log_probs = torch.stack([probs[i][torch.arange(data[:, i].size(0)), data[:, i]].log() for i in range(data.size(-1))]).t()
        n_log_likeli = - log_probs.mean(0).sum()
        return n_log_likeli

    def forward(self, input, data):       
        probs = self.decode(input)
        rec_loss = self.calc_rec_loss(probs, data)
        return rec_loss, probs

class CovarianceMatrix(nn.Module):
    def __init__(self, size, mode='full'):
        super(CovarianceMatrix, self).__init__()
        self.size = size
        self.mode = mode
        self.log_sigma = nn.Parameter(torch.randn(1))        
        
        if mode == 'full':
            self.L_param = nn.Parameter(torch.randn(size, size))
            
        elif mode == 'diagonal':
            self.L_param = nn.Parameter(torch.randn(size))

    def forward(self):
        if self.mode == 'full':
            L = torch.tril(self.L_param)
            Phi = torch.mm(L, L.t())
        
        elif self.mode == 'diagonal':
            Phi = torch.diag(torch.nn.functional.softplus(self.L_param))
            
        return Phi, torch.nn.functional.softplus(self.log_sigma)
    
    
def thermometer_encode_column(df, col):
    df[col] = pd.Categorical(df[col], categories=np.unique(df[col]).astype(int), ordered=True)
    unique_vals = df[col].cat.categories[1:]
    encoded_df = pd.DataFrame()

    for i, category in enumerate(unique_vals):
        encoded_df[f'{col}_{category}'] = (df[col].cat.codes >= i+1).astype(int)

    return encoded_df

def thermometer_encode_df(df, cols):
    encoded_dfs = []
    for col in cols:
        encoded_df = thermometer_encode_column(df, col)
        encoded_dfs.append(encoded_df)

    all_encoded = pd.concat(encoded_dfs, axis=1)
    
    return all_encoded.astype(np.float32)
