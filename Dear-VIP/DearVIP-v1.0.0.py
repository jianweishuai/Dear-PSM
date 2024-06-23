# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:59:14 2024

@author: hqz
"""
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as Data

class DeepNeuralNetwork(nn.Module):
    def __init__(self, x_dim, u1, u2, u3, u4, dropout):
        super(DeepNeuralNetwork, self).__init__()
        
        cat_dim = u1+u2[-1]+u3[-1]+u4[-1]
        
        self.p1_1 = nn.Linear(x_dim, u1)
        
        self.p2_1 = nn.Linear(x_dim, u2[0])
        self.p2_2 = nn.Linear(u2[0], u2[1])
        
        self.p3_1 = nn.Linear(x_dim, u3[0])
        self.p3_2 = nn.Linear(u3[0], u3[1])
        
        self.p4_1 = nn.Linear(x_dim, u4[0])
        self.p4_2 = nn.Linear(u4[0], u4[1])
        self.p4_3 = nn.Linear(u4[1], u4[2])
        
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)
        self.drop4 = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(cat_dim, 1)
        
    def forward(self, x):
        
        p1 = self.drop1(F.relu(self.p1_1(x)))
        
        p2_1 = F.relu(self.p2_1(x))
        p2_2 = self.drop2(F.relu(self.p2_2(p2_1)))
        
        p3_1 = F.relu(self.p3_1(x))
        p3_2 = self.drop3(F.relu(self.p3_2(p3_1)))
        
        p4_1 = F.relu(self.p4_1(x))
        p4_2 = F.relu(self.p4_2(p4_1))
        p4_3 = self.drop4(F.relu(self.p4_3(p4_2)))
        
        h = torch.cat([p1, p2_2, p3_2, p4_3], 1)
        
        out = self.fc(h)
        
        return out
    
class FullyConnectedNetwork(nn.Module):
    def __init__(self, x_dim, u1, u2, u3, dropout):
        super(FullyConnectedNetwork, self).__init__()
        
        self.fc1 = nn.Linear(x_dim, u1)
        self.fc2 = nn.Linear(u1, u2)
        self.fc3 = nn.Linear(u2, u3)
        
        self.fc = nn.Linear(u3, 1)
        
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        
        h = self.drop1(F.relu(self.fc1(x)))
        h = self.drop2(F.relu(self.fc2(h)))
        h = self.drop3(F.relu(self.fc3(h)))
        
        out = self.fc(h)
        
        return out

def TrainDNN(data_set, data_pred, mission):
    
    n_epochs = 300
    batch_size = 512
    
    if torch.cuda.is_available():
        use_device = 'on GPU'
        cuda = True
    else:
        use_device = 'on CPU'
        cuda = False
    
    x_dim = data_set.shape[1] - 1
    n_valids = int(0.1 * len(data_set))
    
    v_batchs = n_valids // batch_size
    n_valids = v_batchs * batch_size
    
    valid_data = data_set[:n_valids]
    train_data = data_set[n_valids:]
    
    torch_data = Data.TensorDataset(torch.Tensor(train_data[:, :-1]), 
                                    torch.Tensor(train_data[:,-1].reshape((-1,1))))
    
    train_iter = Data.DataLoader(dataset=torch_data, 
                                 batch_size=batch_size,
                                 shuffle=True, drop_last=True)
    
    model = DeepNeuralNetwork(x_dim, 128, (192, 256), (32, 64), (192, 64, 16), 0.1)
    
    adam_opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    if mission == 0:
        print('[INFO] Train Deep Neural Network to fit and predict RT ' + use_device)
        loss_fun = torch.nn.MSELoss()
    if mission == 1:
        print('[INFO] Train Deep Neural Network to calculate discriminate score ' + use_device)
        loss_fun = torch.nn.BCEWithLogitsLoss()
    
    if cuda:
        model.cuda()
    
    valid_x = torch.Tensor(valid_data[:,:-1])
    valid_y = torch.Tensor(valid_data[:,-1].reshape((-1,1)))
    
    if cuda:
        valid_x = valid_x.cuda()
        valid_y = valid_y.cuda()
    
    save_train_loss = []
    save_valid_loss = []
    
    for t in range(n_epochs+1):
        
        n = 0
        train_loss = 0
        
        model.train()
        
        for x, y in train_iter:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            
            adam_opt.zero_grad()
            
            pred = model(x)
            
            loss = loss_fun(pred, y)
            
            n += 1
            train_loss += loss.item()
            
            loss.backward()
            adam_opt.step()
        
        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x)
            valid_loss = loss_fun(valid_pred, valid_y)
            valid_loss = valid_loss.item()
        
        save_train_loss.append(str(train_loss/n))
        save_valid_loss.append(str(valid_loss))
        
        if t % 50 == 0:
            print('[INFO] epoch: %d| train_loss: %.6f| valid_loss: %.6f'%(t, train_loss/n, valid_loss))
    
    model.eval()
    
    x = torch.Tensor(data_set[:,:-1])
    if cuda:
        x = x.cuda()
    
    y_set = model(x)
    y_set = y_set.detach().cpu().numpy()
    
    x = torch.Tensor(data_pred)
    if cuda:
        x = x.cuda()
    
    y_pred = model(x)
    y_pred = y_pred.detach().cpu().numpy()
    
    return y_set, y_pred

def TrainFCNetwork(data_set, data_pred, mission):
    
    n_epochs = 300
    batch_size = 512
    
    if torch.cuda.is_available():
        use_device = 'on GPU'
        cuda = True
    else:
        use_device = 'on CPU'
        cuda = False
    
    x_dim = data_set.shape[1] - 1
    n_valids = int(0.1 * len(data_set))
    
    v_batchs = n_valids // batch_size
    n_valids = v_batchs * batch_size
    
    valid_data = data_set[:n_valids]
    train_data = data_set[n_valids:]
    
    torch_data = Data.TensorDataset(torch.Tensor(train_data[:, :-1]), 
                                    torch.Tensor(train_data[:,-1].reshape((-1,1))))
    
    train_iter = Data.DataLoader(dataset=torch_data, 
                                 batch_size=batch_size,
                                 shuffle=True, drop_last=True)
    
    model = FullyConnectedNetwork(x_dim, 512, 512, 512, 0.1)
    
    adam_opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    if mission == 0:
        print('[INFO] Train Deep Neural Network to fit and predict RT ' + use_device)
        loss_fun = torch.nn.MSELoss()
    if mission == 1:
        print('[INFO] Train Deep Neural Network to calculate discriminate score ' + use_device)
        loss_fun = torch.nn.BCEWithLogitsLoss()
    
    if cuda:
        model.cuda()
    
    valid_x = torch.Tensor(valid_data[:,:-1])
    valid_y = torch.Tensor(valid_data[:,-1].reshape((-1,1)))
    
    if cuda:
        valid_x = valid_x.cuda()
        valid_y = valid_y.cuda()
    
    save_train_loss = []
    save_valid_loss = []
    
    for t in range(n_epochs+1):
        
        n = 0
        train_loss = 0
        
        model.train()
        
        for x, y in train_iter:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            
            adam_opt.zero_grad()
            
            pred = model(x)
            
            loss = loss_fun(pred, y)
            
            n += 1
            train_loss += loss.item()
            
            loss.backward()
            adam_opt.step()
        
        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x)
            valid_loss = loss_fun(valid_pred, valid_y)
            valid_loss = valid_loss.item()
        
        save_train_loss.append(str(train_loss/n))
        save_valid_loss.append(str(valid_loss))
        
        if t % 50 == 0:
            print('[INFO] epoch: %d| train_loss: %.6f| valid_loss: %.6f'%(t, train_loss/n, valid_loss))
    
    model.eval()
    
    x = torch.Tensor(data_set[:,:-1])
    if cuda:
        x = x.cuda()
    
    y_set = model(x)
    y_set = y_set.detach().cpu().numpy()
    
    x = torch.Tensor(data_pred)
    if cuda:
        x = x.cuda()
    
    y_pred = model(x)
    y_pred = y_pred.detach().cpu().numpy()
    
    return y_set, y_pred

def RSquareScore(y, y_pred):
    return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)

def DataNormalize(data):
    
    miu = data.mean(0)
    std = data.std(0) + 1e-30
    return (data - miu) / std

def ROCAUC(y, pred):
    
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    
    pred_sort = np.sort(pred)[::-1]
    index = np.argsort(pred)[::-1]
    y_sort = y[index]
    
    tpr = []
    fpr = []
    for i,item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
    
    auc = 0
    last_x = 0
    for x, y in zip(fpr, tpr):
        auc += (x - last_x) * y
        last_x = x
    
    return fpr, tpr, auc

def RemoveDuplicates(raw_data):
    
    raw_data = sorted(raw_data, key=lambda x: x['ModifiedPeptideSequence'])
    
    uniq_pepts = dict()
    for i, line in enumerate(raw_data):
        
        score = line['HyperScore']
        key = line['ModifiedPeptideSequence'] + str(line['PrecursorCharge'])
        
        #key = line['ModifiedPeptideSequence']
        
        if key not in uniq_pepts:
            uniq_pepts[key] = [i, score]
        else:
            if score > uniq_pepts[key][1]:
                uniq_pepts[key] = [i, score]
    
    uniq_data = []
    for value in uniq_pepts.values():
        uniq_data.append(raw_data[value[0]])
    
    uniq_data = sorted(uniq_data, key=lambda x: x['HyperScore'], reverse=True)
    
    return uniq_data

def PeptideFDR(pept_score):
    d = 0
    t = 1e-30
    pept_fdr = []
    
    for i in range(len(pept_score)):
        if pept_score[i][2] == 1:
            d += 1
        else:
            t += 1
        
        pept_fdr.append(d / t * (1e7-t) / (1e7-d))
    
    pept_fdr.reverse()
    
    q_min = 1e30
    for i in range(len(pept_fdr)):
        q_min = min(q_min, pept_fdr[i])
        pept_fdr[i] = q_min
    
    pept_fdr.reverse()
    
    filter_pept = []
    for i in range(len(pept_score)):
        if pept_fdr[i] > 0.01:
            break
        filter_pept.append(pept_score[i])
    
    plot_fdr = [x for x in pept_fdr if x < 0.05]
    
    return filter_pept, plot_fdr

def ProteinFDR(pept_score):
    
    d = 0
    D = 1e10
    prot_score = {}
    for i in range(len(pept_score)):
        if pept_score[i][2] == 1:
            d += 1
        
        if pept_score[i][2] == 1:
            p = (d-0.5) / D
        else:
            p = (d+0.5) / D
        
        LP = -np.log10(p)
        
        prot = pept_score[i][1]
        decoy = pept_score[i][2]
        if prot not in prot_score:
            prot_score[prot] = [LP, 1, decoy]
        else:
            prot_score[prot][1] += 1
            
            if LP > prot_score[prot][0]:
                prot_score[prot][0] = LP
    
    prot_lpgm = []
    for prot, score in prot_score.items():
        n = score[1]
        LPM = score[0]
        decoy = score[2]
        
        LPGM = -np.log10(1- np.power(1-np.power(10, -LPM), n))
        
        prot_lpgm.append([prot, LPGM, decoy])
    
    prot_lpgm = sorted(prot_lpgm, key=lambda x: x[1], reverse=True)
    
    d = 0
    t = 1e-30
    prot_fdr = []
    for i in range(len(prot_lpgm)):
        if prot_lpgm[i][2] == 1:
            d += 1
        else:
            t += 1
        
        prot_fdr.append(d / t * (1e7-t) / (1e7-d))
    
    prot_fdr.reverse()
    
    q_min = 1e30
    for i in range(len(prot_fdr)):
        q_min = min(q_min, prot_fdr[i])
        prot_fdr[i] = q_min
    
    prot_fdr.reverse()
    
    filter_prot = set()
    for i in range(len(prot_lpgm)):
        if prot_fdr[i] > 0.01:
            break
        filter_prot.add(prot_lpgm[i][0])
    
    plot_fdr = [x for x in prot_fdr if x < 0.05]
    
    return filter_prot, plot_fdr

def WriteLibrary(data, out_folder):
    
    with open(out_folder+'library.tsv', 'w') as lib_file:
        
        head = "PrecursorMz\tProductMz\tAnnotation\tProteinId\t" +\
                "GeneName\tPeptideSequence\tModifiedPeptideSequence\t" +\
                "PrecursorCharge\tLibraryIntensity\t" +\
                "FragmentType\tFragmentCharge\tFragmentSeriesNumber\t" +\
                "AverageExperimentalRetentionTime\n"
        
        lib_file.write(head)
        
        for n, line in enumerate(data):
            precursorMz = str(line['PrecursorMz'])
            proteinId = line['ProteinId']
            
            gene_names = ""
            if ';' in proteinId:
                spl_prot = proteinId.split(';')
                for s in spl_prot:
                    gene_names += s.split('|')[1] + ';'
                gene_names = gene_names[:-1]
            else:
                gene_names = proteinId.split('|')[1]
                
            pept_seq = line['PeptideSequence']
            mod_seq = line['ModifiedPeptideSequence']
            charge = str(int(line['PrecursorCharge']))
            
            avg_rt = str(line['ExperimentalRetentionTime'])
            
            productMz = line['ProductMz'].split(';')
            annotation = line['Annotation'].split(';')
            lib_inten = line['LibraryIntensity'].split(';')
            
            out_str = ""
            for i in range(len(productMz)):
                series = annotation[i].split('^')[0]
                series = series[1:]
                
                out_str += precursorMz + "\t" + \
                           productMz[i] + "\t" + \
                           annotation[i] + "\t" + \
                           proteinId + "\t" + \
                           gene_names + "\t" + \
                           pept_seq + "\t" + \
                           mod_seq + "\t" + \
                           charge + "\t" + \
                           lib_inten[i] + "\t" + \
                           annotation[i][0] + "\t" + \
                           annotation[i][-1] + "\t" + \
                           series + "\t" + \
                           str(avg_rt) + "\n"
            
            lib_file.write(out_str)

def PredictRetentionTime(data):
    
    n_aa = 26
    x_dim = 3 * n_aa + 3
    
    exp_rt = np.zeros((len(data), 1)) # experimental retention time
    pred_x = np.zeros((len(data), x_dim))
    
    n_fit = 0
    fit_y = np.zeros((len(data), 1))
    fit_x = np.zeros((len(data), x_dim))
    
    n_decoys = 0
    n_targets = 0
    
    for i, line in enumerate(data):
        
        mass = line['PeptideMass']
        raw_seq = line['PeptideSequence']
        rt = line['ExperimentalRetentionTime']
        
        exp_rt[i] = rt
        
        for j, s in enumerate(raw_seq):
            idx = ord(s) - ord('A')
            pred_x[i][idx] += 1
            
            if j == 0 or j == 1:
                pred_x[i][n_aa + idx] += 1
            if j == len(raw_seq)-1 or j == len(raw_seq)-2:
                pred_x[i][2*n_aa + idx] += 1
        
        pred_x[i][x_dim-3] = len(raw_seq)
        pred_x[i][x_dim-2] = np.log1p(mass)
        pred_x[i][x_dim-1] = 1
        
        if line['Decoy'] == 0:
            n_targets += 1
            
            if n_decoys / n_targets < 0.01:
                fit_y[n_fit] = exp_rt[i]
                fit_x[n_fit] = pred_x[i]
                n_fit += 1
        else:
            n_decoys += 1
    
    fit_x = fit_x[:n_fit]
    fit_y = fit_y[:n_fit]
    
    fit_x = DataNormalize(fit_x)
    pred_x = DataNormalize(pred_x)
    
    if len(fit_x) < 1000:
        x_T = fit_x.T
        cov = x_T.dot(fit_x)
        b = x_T.dot(fit_y)
        
        z = np.linalg.pinv(cov) @ b
        
        fit_rt = fit_x.dot(z)
        pred_rt = pred_x.dot(z)
    else:
        
        min_y = np.min(exp_rt)
        max_y = np.max(exp_rt)
        
        fit_set = np.c_[fit_x, fit_y]
        np.random.shuffle(fit_set)
        
        fit_set[:,-1] = (fit_set[:,-1] - min_y) / (max_y - min_y)
        
        fit_rt, pred_rt = TrainDNN(fit_set, pred_x, 0)
        
        fit_y = fit_set[:,-1] * (max_y - min_y) + min_y
        fit_rt = fit_rt * (max_y - min_y) + min_y
        pred_rt = pred_rt * (max_y - min_y) + min_y
        
    return fit_y.flatten(), fit_rt.flatten(), pred_rt.flatten()
    

class LDA:
    def __init__(self):
        self.w = None
    
    def fit(self, x, y):
        
        x1 = x[y.flatten() == 0]
        x2 = x[y.flatten() == 1]
        
        if torch.cuda.is_available():
            
            print('[INFO] Use Linear Discriminant Analysis to fit and predict RT on GPU')
            
            x1 = torch.Tensor(x1)
            x2 = torch.Tensor(x2)
            
            x1 = x1.cuda()
            x2 = x2.cuda()
            
            miu1 = torch.mean(x1, dim=0)
            miu2 = torch.mean(x2, dim=0)
            
            cov1 = torch.matmul((x1 - miu1).T, (x1 - miu1))
            cov2 = torch.matmul((x2 - miu2).T, (x2 - miu2))
            Sw = cov1 + cov2
            
            self.w = torch.matmul(torch.linalg.pinv(Sw), (miu1 - miu2).view((-1, 1)))
            
        else:
            
            print('[INFO] Use Linear Discriminant Analysis to fit and predict RT on CPU')
            
            miu1 = np.mean(x1, axis=0)
            miu2 = np.mean(x2, axis=0)
            
            cov1 = np.dot((x1 - miu1).T, (x1 - miu1))
            cov2 = np.dot((x2 - miu2).T, (x2 - miu2))
            Sw = cov1 + cov2
        
            self.w = np.dot(np.linalg.pinv(Sw), (miu1 - miu2).reshape((-1, 1)))
    
    def predict(self, x):
        if torch.cuda.is_available():
            x = torch.Tensor(x)
            x = x.cuda()
            y = torch.matmul((x), self.w)
            
            return y.detach().cpu().numpy()
        else:
            return np.dot((x), self.w)

def DiscriminateScore(data, pred_rt):
    
    x_dim = 10
    
    real_y = np.zeros((len(data), 1))
    pred_x = np.zeros((len(data), x_dim))
    
    fit_y = np.zeros((len(data), 1))
    fit_x = np.zeros((len(data), x_dim))
    
    is_decoy = []
    
    for i, line in enumerate(data):
        
        decoy = line['Decoy']
        match_ions = line['MatchIons']
        miss_cleavage = line['MissCleavage']
        charge = line['PrecursorCharge']
        mass_diff = np.abs(line['MassDiff']) + 1e-100
        seq_len = len(line['PeptideSequence'])
        inten_ratio = line['IntensityRatio']
        rt = line['ExperimentalRetentionTime']
        poisson_score = line['PoissonScore'] + 1e-100
        
        hyperscore = line['HyperScore']
        xcorr_score = line['XcorrScore']
        
        is_decoy.append(decoy)
        real_y[i] = 1 - decoy
        
        pred_x[i] = np.array([
            np.sqrt(np.abs(rt - pred_rt[i])),
            np.log1p(mass_diff),
            poisson_score,
            np.log1p(inten_ratio),
            np.log1p(charge),
            np.log1p(match_ions),
            np.log1p(seq_len),
            np.log1p(hyperscore),
            np.log1p(xcorr_score),
            np.log1p(miss_cleavage)
            ])
    
    n_fit = 0
    n_decoys = 0
    n_targets = 0
    
    n_t = 0
    n_d = 0
    
    tag_pepts = set()
    
    for i, line in enumerate(data):
        decoy = line['Decoy']
        
        if decoy == 0:
            n_targets += 1
            
            if n_decoys / n_targets < 0.01:
                fit_y[n_fit] = real_y[i]
                fit_x[n_fit] = pred_x[i]
                
                tag_pepts.add(data[i]['ModifiedPeptideSequence'])
                
                n_t += 1
                
                n_fit += 1
        else:
            n_decoys += 1
            
        if decoy == 1:
            fit_y[n_fit] = real_y[i]
            fit_x[n_fit] = pred_x[i]
            n_fit += 1
            
            n_d += 1
            
    fit_x = fit_x[:n_fit]
    fit_y = fit_y[:n_fit]
    
    fit_x = DataNormalize(fit_x)
    pred_x = DataNormalize(pred_x)
    
    if len(fit_x) < 1000:
        model = LDA()
        
        model.fit(fit_x, fit_y)
        pred_y = model.predict(pred_x)
        fit_pred = model.predict(fit_x)
        
    else:
        fit_set = np.c_[fit_x, fit_y]
        np.random.shuffle(fit_set)
        
        fit_x = fit_set[:, :-1]
        fit_y = fit_set[:, -1].reshape((-1,1))
        
        fit_pred, pred_y = TrainFCNetwork(fit_set, pred_x, 1)
        
    decoy_score = []
    target_score = []
    for i in range(len(is_decoy)):
        if is_decoy[i] == 1:
            decoy_score.append(pred_y[i])
        else:
            target_score.append(pred_y[i])
    
    return target_score, decoy_score, pred_y


def DearVIP(file_name, fdr_type):
    
    #Step0: Load raw data
    read_data = []
    with open(file_name, 'r') as data_file:
        
        line_count = 0
        for line in data_file:
            line_count += 1
            
            if line_count == 1:
                columns = line.split() + ['DiscriminatedScore']
            else:
                line = line.split() + [0]
                read_data.append(line)
    
    raw_data = []
    for i in range(len(read_data)):
        line = dict()
        for j, col in enumerate(columns):
            try:
                x = float(read_data[i][j])
            except:
                x = read_data[i][j]
                
            line[col] = x
        
        raw_data.append(line)
    
    # Step1: Remove duplicate precursor ions
    data = RemoveDuplicates(raw_data)
    
    figure_data = dict()
    
    #Step2: Predict Retention Time
    fit_y, fit_rt, pred_rt = PredictRetentionTime(data)
    
    #Step3: Calculate Discriminated Score
    target_score, decoy_score, dscore = DiscriminateScore(data, pred_rt)
    
    figure_data['target_score'] = target_score
    figure_data['decoy_score'] = decoy_score
    
    for i in range(len(data)):
        data[i]['DiscriminatedScore'] = dscore[i]
            
    #Step4: Calculate FDR
    data = sorted(data, key=lambda x: x['ModifiedPeptideSequence'])
    
    mod_pepts = {}
    for line in data:
        
        prot = line['ProteinId']
        decoy = line['Decoy']
        score = line['DiscriminatedScore']
        key = line['ModifiedPeptideSequence']
        
        if key not in mod_pepts:
            mod_pepts[key] = [prot, decoy, score]
        else:
            if score > mod_pepts[key][-1]:
                mod_pepts[key] = [prot, decoy, score]
    
    mod_pepts = sorted(mod_pepts.items(), key=lambda x: x[1][-1], reverse=True)
    
    pept_score = []
    for i in range(len(mod_pepts)):
        pept_score.append([mod_pepts[i][0]] + mod_pepts[i][1])
    
    filter_pepts, plot_fdr = PeptideFDR(pept_score)
    
    figure_data['pept_fdr'] = plot_fdr
    
    filter_prots, plot_fdr = ProteinFDR(pept_score)
    
    figure_data['prot_fdr'] = plot_fdr
    
    #Step5: Filter Peptides and Proteins
    lib_pepts = set()
    for line in filter_pepts:
        if fdr_type == "prot":
            if line[1] in filter_prots and line[2] == 0:
                lib_pepts.add(line[0])
        if fdr_type == "pept":
            if line[2] == 0:
                lib_pepts.add(line[0])
    
    add_pepts = set()
    
    lib_data = []
    for line in data:
        seq = line['ModifiedPeptideSequence']
        if seq in lib_pepts:
            lib_data.append(line)
            
            add_pepts.add(seq)
    
    #Step6: Normalize Retention Time
    out_folder = '/'.join(file_name.split('/')[:-1]) + '/'
    
    WriteLibrary(lib_data, out_folder)
    
    out_folder = '/'.join(file_name.split('/')[:-1]) + '/'
    
    with open(out_folder+'validation_data.pkl', 'wb') as file:
        pickle.dump(figure_data, file)

if __name__ == "__main__":
    
    file = input("Please enter the path of the Dear-PSM output file below:\n")
    file = file.replace("\\", "/")
    
    OK = os.path.exists(file) and os.path.isfile(file)
    
    while OK == False:
        print("Cannot open Dear-PSM output file. Please check your file or directory!\n")
        
        file = input("Please enter the path of the Dear-PSM output file below:\n")
        file = file.replace("\\", "/")
        
        OK = os.path.exists(file) and os.path.isfile(file)
        
    fdr_type = ""
    while fdr_type != "prot" and fdr_type != "pept":
        fdr_type = input("Please enter the fdr type below, choose \"prot\" or  \"pept\":\n")
    
    DearVIP(file, fdr_type)
    