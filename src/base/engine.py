import os
import time
import torch
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from src.models.__init__ import *
from src.utils.metrics import masked_mape,masked_rmse,compute_all_metrics
import ipdb

import wandb
wandb.init(project="st-lora")

class BaseEngine():
    def __init__(self, args):
       # def __init__(self, device, model, dataloader, scaler, sampler, loss_fn, lrate, optimizer, \
    #              scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed):
        super().__init__()
        self._device = args.device
        try:
            self.model = args.engine.model
        except:
            self.model = args.model
            
        self.model.to(self._device)

        self.args = args
        self._dataloader = args.dataloader
        self._scaler = args.scaler

        self._loss_fn = args.loss_fn
        self._lrate = args.lrate
        self._optimizer = args.optimizer
        self._lr_scheduler = args.scheduler
        self._clip_grad_value = args.clip_grad_value

        self._max_epochs = args.max_epochs
        self._patience = args.patience
        self._iter_cnt = 0
        self._save_path = args.log_dir
        self._logger = args.logger
        self._seed = args.seed
        self.save_interval = args.save_interval

        try:
            self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 
        except:
            print("None param_num function")

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path, filename = None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if filename == None:
            filename = 'final_model_s{}.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path, filename=None):
        if filename == None:
            filename = 'final_model_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(os.path.join(save_path, filename)))   


    def train_batch(self,epoch=None):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            
            device = torch.device(self._device)
            
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])
            
            # handle the precision issue when performing inverse transform to label
            # if not epoch == None:
            #    torch.save(pred, f'./save/pred_epoch_{epoch}.pt')
            #    print("pred:",pred.shape,"epoch:",epoch,"save!")
            #==========================

#            mask_value = torch.tensor(0)
            mask_value = torch.tensor(0.1, device=label.device)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            # Check gradient
            for name, param in self.model.named_parameters():
                if param.grad is None:
#                    print(f"Gradient for {name} is None")
                    pass
                else:
#                    print(f"Gradient for {name}: {param.grad.mean()}")
                    pass
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')
        wait = 0
        min_loss = np.inf
        for epoch in tqdm(range(self._max_epochs)):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch(epoch)
            t2 = time.time()

            wandb.log({'epoch': epoch, 'mae': mtrain_loss, 'mape':mtrain_mape, 'rmse':mtrain_rmse})

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._optimizer.param_groups[0]['lr']
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

            if epoch % self.save_interval == 0:
                fn = 'ep{}_s{}.pt'.format(epoch, self._seed)
                self.save_model(self._save_path,filename=fn)

        self._logger.info('best valid_loss:{:.6f}'.format(min_loss)) #
        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                if not os.path.exists('./save'):
                    os.makedirs('./save')

                if self.args.save is not None:
                    torch.save(pred, './save/pred_{}'.format(self.args.save))
                    torch.save(label, './save/label_{}'.format(self.args.save))

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                wandb.log({'Horizon': i+1, 'Test MAE': res[0], 'Test RMSE': res[2], 'Test MAPE': res[1]})

                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])
    
            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
            wandb.log({'Average Test MAE': np.mean(test_mae), 'Test RMSE': np.mean(test_rmse), 'Test MAPE': np.mean(test_mape)})
        
        #ipdb.set_trace()

            # 方便记录结果
            results_name = "results"+str(self._seed)+".csv"
            with open(results_name, 'w', newline='') as csvfile:
                fieldnames = ['MAE3', 'RMSE3', 'MAPE3','MAE6', 'RMSE6', 'MAPE6','MAE12', 'RMSE12', 'MAPE12','MAEa', 'RMSEa', 'MAPEa']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                tmp_res = []
                for i in range(self.model.horizon):
                    if (i+1)%3==0 and (i+1)!=9:
                        tmp_res.append( round(test_mae[i],4))
                        tmp_res.append( round(test_rmse[i],4))
                        tmp_res.append( round(test_mape[i],4))
                tmp_res.append( round(np.mean(test_mae),4))
                tmp_res.append( round(np.mean(test_rmse),4))
                tmp_res.append( round(np.mean(test_mape),4))
                writer.writerow(dict(zip(fieldnames, tmp_res)))