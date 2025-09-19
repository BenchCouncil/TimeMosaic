from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import torch.nn.functional as F
import copy


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')


class Exp_BLAST(Exp_Basic):
    def __init__(self, args):
        super(Exp_BLAST, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true).item()

                total_loss.append(loss)

        total_loss = float(np.mean(total_loss))
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                    
                outputs, cls_soft = self.model(batch_x)

                # cls_soft = self.model.patch_embedding.latest_cls_soft  # [N, num_classes]
                current_ratio = cls_soft.mean(dim=0)
                    
                target_ratio = torch.full_like(current_ratio, 1.0 / len(current_ratio)).detach()
                loss_reg = criterion(current_ratio, target_ratio)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                lambda_cls = 0.001
                loss = criterion(outputs, batch_y) + lambda_cls * loss_reg

                train_loss.append(loss.mean().item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.mean().item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                
                # loss.backward()
                loss.mean().backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            mse96,  mae96  = self.test_blast(setting, 96)
            mse192, mae192 = self.test_blast(setting, 192)
            mse336, mae336 = self.test_blast(setting, 336)
            mse720, mae720 = self.test_blast(setting, 720)

            print(
                f"[TEST] {self.args.test_data_path}: \n"
                f"96(MSE={mse96:.6f}, MAE={mae96:.6f}), \n"
                f"192(MSE={mse192:.6f}, MAE={mae192:.6f}), \n"
                f"336(MSE={mse336:.6f}, MAE={mae336:.6f}), \n"
                f"720(MSE={mse720:.6f}, MAE={mae720:.6f}) \n",
                flush=True
            )
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | Test MSE: {4:.7f} Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, mse96, mae96))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test_blast(self, setting, test_pred_len):
        args_blast = copy.deepcopy(self.args)

        args_blast.root_path = args_blast.test_root_path
        args_blast.data_path = args_blast.test_data_path
        args_blast.data = args_blast.test_data
        args_blast.pred_len = test_pred_len

        test_data, test_loader = data_provider(args_blast, 'test')
        
        preds = []
        trues = []

        self.model.eval()
        all_pred_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs  = self.model(batch_x, args_blast.pred_len // 16)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        mae, mse, rmse, mape, mspe, _ = metric(preds, trues)
        self.model.train()

        return mse, mae


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

            if any(k.startswith("module.") for k in state_dict.keys()):
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict

            self.model.load_state_dict(new_state_dict, strict=False)



        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        all_pred_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs  = self.model(batch_x, self.args.pred_len // 16)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # if cls_pred is not None:
                #     all_pred_list.append(cls_pred.cpu())


                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #    # if getattr(test_data, "scale", False) and getattr(self.args, "inverse", False):
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # print(len(all_pred_list))
        # if self.args.counts:
        #     if len(all_pred_list) > 0:
        #         patch_len_list = eval(self.args.patch_len_list)
        #         all_cls_pred = torch.cat(all_pred_list)
        #         patch_counts = torch.bincount(all_cls_pred, minlength=len(patch_len_list))
        #         print("Granularity counts:")
        #         for i, patch_len in enumerate(patch_len_list):
        #             print(f"Patch size {patch_len}: {patch_counts[i].item()} 次")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, _ = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        
        self.profile_model(test_loader)

        return
    
    def profile_model(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()

            _ = self.model(batch_x)

            torch.cuda.synchronize()
            end_time = time.time()

            inference_time = end_time - start_time
            gpu_mem = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print("=" * 80)
            print("Model Profiling Summary")
            print(f"{'Total Params':<25}: {total_params:,}")
            print(f"{'Inference Time (s)':<25}: {inference_time:.6f}")
            print(f"{'GPU Mem Footprint (MB)':<25}: {gpu_mem:.2f}")
            print(f"{'Peak Mem (MB)':<25}: {peak_mem:.2f}")
            print("=" * 80)
