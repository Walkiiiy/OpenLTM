import os
import time
import json
import shutil
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, EarlyStoppingHitMAE, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        self.best_val_metrics = None
        self.best_test_metrics = None

    def _get_output_dir(self):
        user_id = self.args.user_id if self.args.user_id else self.args.model_id
        return os.path.join(self.args.output_root, user_id)

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], batch[3], None, None
        if len(batch) == 5:
            return batch[0], batch[1], batch[2], batch[3], batch[4], None
        if len(batch) >= 6:
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        raise ValueError("Unexpected batch format")

    def _compute_hit20_mae(self, preds_min, trues_min):
        abs_err = np.abs(preds_min - trues_min)
        hit20 = np.mean(abs_err <= 20)
        mae = np.mean(abs_err)
        return hit20, mae
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            if self.args.use_cpu or not torch.cuda.is_available():
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(f'cuda:{self.args.gpu}')
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            model = model.to(self.device)
            
        if self.args.adaptation:
            checkpoint = torch.load(self.args.pretrain_model_path)
            if self.args.data == 'LoginIntervalUser':
                model_state = model.state_dict()
                filtered = {
                    k: v for k, v in checkpoint.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                model_state.update(filtered)
                model.load_state_dict(model_state, strict=False)
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print(f"Loaded {len(filtered)}/{len(model_state)} parameters from pretrain checkpoint")
            else:
                model.load_state_dict(checkpoint)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        preds = []
        trues = []
        
        self.model.eval()    
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                iter_count += 1
                batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])

                if self.args.data == 'LoginIntervalUser':
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        if self.args.data == 'LoginIntervalUser':
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(preds.shape[0], -1)
            trues = trues.reshape(trues.shape[0], -1)
            preds_min = vali_data.inverse_transform_y(preds).reshape(-1)
            trues_min = vali_data.inverse_transform_y(trues).reshape(-1)
            hit20, mae = self._compute_hit20_mae(preds_min, trues_min)
            return total_loss, hit20, mae

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        if self.args.data == 'LoginIntervalUser':
            early_stopping = EarlyStoppingHitMAE(self.args, verbose=True)
            output_dir = self._get_output_dir()
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                os.makedirs(output_dir, exist_ok=True)
                log_path = os.path.join(output_dir, 'train_log.txt')
        else:
            early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if self.args.data == 'LoginIntervalUser':
                vali_loss, val_hit20, val_mae = self.vali(
                    vali_data, vali_loader, criterion, is_test=self.args.valid_last)
                test_loss, test_hit20, test_mae = self.vali(
                    test_data, test_loader, criterion, is_test=True)
                self.best_test_metrics = {'hit20': float(test_hit20), 'mae': float(test_mae)}
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print(
                        "Epoch: {}, Steps: {} | Vali Loss: {:.7f} Hit@20: {:.4f} MAE: {:.4f} | Test Loss: {:.7f} Hit@20: {:.4f} MAE: {:.4f}".format(
                            epoch + 1, train_steps, vali_loss, val_hit20, val_mae, test_loss, test_hit20, test_mae
                        )
                    )
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(
                            f"epoch={epoch+1}, val_loss={vali_loss:.7f}, val_hit20={val_hit20:.4f}, val_mae={val_mae:.4f}, "
                            f"test_loss={test_loss:.7f}, test_hit20={test_hit20:.4f}, test_mae={test_mae:.4f}\n"
                        )
                early_stopping(val_hit20, val_mae, self.model, path)
                if self.best_val_metrics is None or val_hit20 > self.best_val_metrics['hit20'] or (
                    abs(val_hit20 - self.best_val_metrics['hit20']) <= 1e-12 and val_mae < self.best_val_metrics['mae']
                ):
                    self.best_val_metrics = {'hit20': float(val_hit20), 'mae': float(val_mae)}
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
                test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                        epoch + 1, train_steps, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        if self.args.data == 'LoginIntervalUser':
            output_dir = self._get_output_dir()
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(best_model_path, os.path.join(output_dir, 'best_checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)
            
        preds = []
        trues = []
        row_indices = []
        timestamps = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                iter_count += 1
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_idx, batch_ts = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                inference_steps = self.args.test_pred_len // self.args.output_token_len
                dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):  
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.output_token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-self.args.output_token_len+dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if self.args.data == 'LoginIntervalUser' and batch_idx is not None:
                    if torch.is_tensor(batch_idx):
                        row_indices.extend(batch_idx.cpu().numpy().tolist())
                    else:
                        row_indices.extend(list(batch_idx))
                    if batch_ts is not None:
                        timestamps.extend(list(batch_ts))
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                if self.args.visualize and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        if self.args.data == 'LoginIntervalUser':
            preds = preds.reshape(preds.shape[0], -1)
            trues = trues.reshape(trues.shape[0], -1)
            preds_min = test_data.inverse_transform_y(preds).reshape(-1)
            trues_min = test_data.inverse_transform_y(trues).reshape(-1)
            hit20, mae = self._compute_hit20_mae(preds_min, trues_min)
            print('Hit@20:{}, MAE:{}'.format(hit20, mae))

            output_dir = self._get_output_dir()
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                os.makedirs(output_dir, exist_ok=True)
                abs_err = np.abs(preds_min - trues_min)
                hit_flag = (abs_err <= 20).astype(int)
                pred_path = os.path.join(output_dir, 'pred_test.csv')
                with open(pred_path, 'w', encoding='utf-8') as f:
                    f.write("row_index,timestamp,true_min,pred_min,abs_error,hit20_flag\n")
                    for i in range(len(preds_min)):
                        idx_val = row_indices[i] if i < len(row_indices) else ''
                        ts_val = timestamps[i] if i < len(timestamps) else ''
                        f.write(f"{idx_val},{ts_val},{trues_min[i]:.6f},{preds_min[i]:.6f},{abs_err[i]:.6f},{hit_flag[i]}\n")

                if self.best_val_metrics is None:
                    val_data, val_loader = self._get_data(flag='val')
                    criterion = self._select_criterion()
                    _, val_hit20, val_mae = self.vali(val_data, val_loader, criterion, is_test=False)
                    self.best_val_metrics = {'hit20': float(val_hit20), 'mae': float(val_mae)}

                metrics_out = {
                    "val": self.best_val_metrics,
                    "test": {"hit20": float(hit20), "mae": float(mae)}
                }
                metrics_path = os.path.join(output_dir, 'metrics.json')
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_out, f, indent=2)
        else:
            if self.args.covariate:
                preds = preds[:, :, -1]
                trues = trues[:, :, -1]
            mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()
        return
