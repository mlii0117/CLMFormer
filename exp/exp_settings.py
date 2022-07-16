from data.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, Transformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

def curriculum_p(p, epoch,gamma):
    return min(p,1-((1.-p) * np.exp(-gamma*epoch) + p))



class Exp_settings(Exp_Basic):
    def __init__(self, args):
        super(Exp_settings, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
             'Transformer': Transformer
        }
        if self.args.model=='informer' or self.args.model=='Transformer':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device,
                self.args.rm_num_slots,
                self.args.rm_d_model,
                self.args.rm_num_heads,path="data_"+str(self.args.data)+"_pl_"+str(self.args.pred_len)+"_el_"+str(self.args.enc_in)+"_dl_"+str(self.args.dec_in)+"_")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            memory = torch.load('./memory_file/'+"data_"+str(self.args.data)+"_pl_"+str(self.args.pred_len)+"_el_"+str(self.args.enc_in)+"_dl_"+str(self.args.dec_in)+"_"+'memory.pt')
            memory=memory.repeat(batch_x.size(0), 1, 1)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        p=0

        memory = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                if memory != None:
                    #print('batch_size', batch_x.size())
                    memory = torch.load('./memory_file/'+"data_"+str(self.args.data)+"_pl_"+str(self.args.pred_len)+"_el_"+str(self.args.enc_in)+"_dl_"+str(self.args.dec_in)+"_"+'memory.pt')
                    #print('memory_before', memory.size())
                    memory=memory.repeat(batch_x.size(0), 1, 1)
                    #print('memory_later', memory.size())


                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)

                        f_dim = -1 if self.args.features=='MS' else 0
                        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)

                    f_dim = -1 if self.args.features=='MS' else 0

                    memory=True





                    batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % self.args.dropout_num == 0:
                    if self.args.curriculum == 1:
                        p += 1
                        for name, child in self.model.named_children():
                            for a, b in child.named_children():
                                if a == "dropout":
                                    b.p = curriculum_p(self.args.dropout_lim, p, self.args.gamma)
                        print('curriculum dropout:', curriculum_p(self.args.dropout_lim, p, self.args.gamma))
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            memory = torch.load('./memory_file/'+"data_"+str(self.args.data)+"_pl_"+str(self.args.pred_len)+"_el_"+str(self.args.enc_in)+"_dl_"+str(self.args.dec_in)+"_"+'memory.pt')
            memory = memory.repeat(batch_x.size(0), 1, 1)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            memory = torch.load('./memory_file/'+"data_"+str(self.args.data)+"_pl_"+str(self.args.pred_len)+"_el_"+str(self.args.enc_in)+"_dl_"+str(self.args.dec_in)+"_"+'memory.pt')
            memory = memory.repeat(batch_x.size(0), 1, 1)


            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,memory)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return


