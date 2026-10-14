from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LGT
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from layers.network import ViewDiagnostics, CoalitionFusion, subset_name

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch import optim
import os
import time
import copy
import json
import warnings
import math
import psutil

def test_model_size(model, filename='temp_model.pt'):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 ** 2)
    print(f"[MODEL SIZE] Model file size: {size_mb:.2f} MB")
    os.remove(filename)

def test_params_memory(model, input_shape):
    # Tính số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PARAMS] Total parameters: {total_params:,}")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)


    def _build_model(self):
        model_dict = {
            'LGT': LGT,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, patience=5)
        return model_optim

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def _wrapper(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _load_pretrained(self):
        """Transfer: --pretrained <path to model.pth trained on another data set>. Any number of channels;
        only the per-channel RevIN calibration is re-initialised (Proposition 2)."""
        pre = getattr(self.args, 'pretrained', None)
        if not pre:
            return
        state = torch.load(pre, map_location=self.device)
        wrapper = self._wrapper()
        if hasattr(wrapper, 'load_transfer'):
            skipped = wrapper.load_transfer(state)
            print(f'[TRANSFER] loaded {pre}; re-initialised: {skipped if skipped else "nothing"}')
        else:
            wrapper.load_state_dict(state)
            print(f'[TRANSFER] loaded {pre}')

    def vali(self, vali_data, vali_loader, criterion, is_test = True):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # calendar of input / target timestamps only (no target values -> no leakage)
                result = self.model(batch_x, x_mark=batch_x_mark, y_mark=batch_y_mark)
                if isinstance(result, tuple):
                    outputs, _ = result
                else:
                    outputs = result
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if not is_test:
                    pred = outputs
                    true = batch_y
                else:
                    pred = outputs#.detach().cpu()
                    true = batch_y#.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # transfer from another data set (optional); train_epochs = 0 -> zero-shot evaluation
        self._load_pretrained()

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        mse_criterion, mae_criterion = self._select_criterion()
        epoch_times = []



        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # train_time = 0 # For computational cost analysis

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # result = self.model(batch_x)
                result = self.model(batch_x, current_epoch=epoch, max_epochs=self.args.train_epochs,
                                    x_mark=batch_x_mark, y_mark=batch_y_mark)
                if isinstance(result, tuple):
                    outputs, aux_losses = result
                else:
                    outputs, aux_losses = result, {}
                # train_time += time.time() - temp # For computational cost analysis
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs
                batch_y = batch_y

                loss = mae_criterion(outputs, batch_y)          # the only training loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 2)
                print(f"[TRAIN MEMORY] Max memory allocated in epoch {epoch+1}: {mem:.2f} MB")
                torch.cuda.reset_peak_memory_stats(device=self.device)
            else:
                print("[TRAIN MEMORY] CUDA not available, cannot measure GPU memory.")

            epoch_duration = time.time() - epoch_time
            epoch_times.append(epoch_duration)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion) # For MSE criterion
            # test_loss = self.vali(test_data, test_loader, critericoston) # For MSE criterion
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            self.scheduler.step(vali_loss)
            print("learning rate: ", self.scheduler.get_last_lr())
            # adjust_learning_rate_new(model_optim, epoch + 1, self.args)

            # print('Alpha:', self.model.decomp.ma.alpha) # Print the learned alpha
            # print('Beta:', self.model.decomp.ma.beta)   # Print the learned beta

        # print("Training time: {}".format(np.sum(train_times)/len(train_times))) # For computational cost analysis
        if epoch_times:
            avg_epoch_time = np.mean(epoch_times)
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        else:
            print("No training epoch (zero-shot evaluation of the loaded weights)")

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
        # persistent copy of the best weights (source model for transfer, reload with test=1)
        torch.save(self._wrapper().state_dict(), os.path.join(path, 'model.pth'))

        return self.model
    def _refit_coalitions(self, epochs=None, lr=1e-2):
        """Fit one fusion per subset of views on the validation split, with the trained views frozen
        (evaluation only; the model itself is not changed). Returns a CoalitionFusion."""
        epochs = int(getattr(self.args, 'coal_epochs', 5)) if epochs is None else epochs
        wrapper = self._wrapper()
        cf = CoalitionFusion(wrapper.net).to(self.device)
        opt = optim.Adam(cf.parameters(), lr=lr)
        _, vali_loader = self._get_data(flag='val')
        flags = [p.requires_grad for p in wrapper.parameters()]
        for p in wrapper.parameters():
            p.requires_grad_(False)
        self.model.eval()
        for ep in range(epochs):
            tot, cnt = 0.0, 0
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                pred = wrapper.coalition_forecasts(cf, batch_x, y_mark=batch_y_mark)       # [B, 7, H, C]
                loss = (pred - batch_y[:, None]).abs().mean(dim=(0, 2, 3)).sum()            # MAE of each subset
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() / 7
                cnt += 1
            print('[COALITIONS] refit epoch %d | mean validation MAE over the 7 subsets %.5f' % (ep + 1, tot / max(cnt, 1)))
        for p, f in zip(wrapper.parameters(), flags):
            p.requires_grad_(f)
        cf.eval()
        return cf


    # ------------------------------------------------------------------------------------------
    # Printing and saving
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _print_game(a, title, loss='MAE'):
        names = ['temporal', 'spectral', 'channel']
        print('\n--- %s ---' % title)
        print('v(S) = %s(window mean) - %s(S) | baseline %s %.5f | v(all) %.5f'
              % (loss, loss, loss, a['loss_baseline'], float(a['value'][7])))
        print('%s of each subset:' % loss,
              {subset_name(m): round(float(a['loss_subset'][m]), 5) for m in range(1, 8)})
        for i, nm in enumerate(names):
            print('%-9s Shapley %.5f (%5.1f%%) | alone v=%.5f | unique %.5f'
                  % (nm, a['shapley'][i], 100 * a['shapley_share'][i], a['value'][1 << i], a['unique'][i]))
        print('interaction index (<0 shared information, >0 synergy):\n', a['interaction'].numpy())
        print('pair  v(i)      v(j)      v(ij)     overlap   gain over better member')
        for k, d in a['pairs'].items():
            print('%-4s  %.5f  %.5f  %.5f  %.5f  %.5f (better: %s)'
                  % (k, d['v_i'], d['v_j'], d['v_ij'], d['overlap'], d['gain_over_better'], d['better']))
        print('(overlap > 0 with v(i), v(j) > 0: the two views share correct information; gain > 0: complementary)')
        g = a['fusion_gain']
        print('all three views over the best single view (%s): %.5f | over the best pair (%s): %.5f'
              % (g['best_single'], g['over_best_single'], g['best_pair'], g['over_best_pair']))

    @staticmethod
    def _print_diagnostics(r):
        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        print('\n--- Views on the test set (data space) ---')
        print('                 temporal  spectral  channel')
        print('MAE per view   :', r['mae_view'].numpy(), '| fused: %.5f' % r['mae_fused'])
        print('MSE per view   :', r['mse_view'].numpy(), '| fused: %.5f' % r['mse_fused'])
        print('mean weight    :', r['mean_weight'].numpy())
        print('Jensen gain    : %.5f  (MAE(fused) = sum_v w_v MAE(view_v) - Jensen gain)' % r['jensen_gain'])
        print('error corr     :\n', r['error_corr'].numpy())
        if 'cross_gain' in r:
            print('cross-channel  : MAE without the other channels %.5f -> with them %.5f (gain %.5f, exact counterfactual)'
                  % (r['mae_without_cross'], r['mae_fused'], r['cross_gain']))
        print('win rate of fused:', r['win_rate'].numpy(), '(share of points where fused is closer than the view)')
        print('oracle (best view at every point) MAE %.5f | fused %.5f' % (r['mae_oracle'], r['mae_fused']))
        if 'refit' in r:
            a = r['refit']
            Exp_Main._print_game(a, 'Shapley, fusion re-fitted for each subset of views (main, MAE)')
            sh = a['shapley_horizon'].numpy()                                         # [3, H]
            q = np.array_split(np.arange(sh.shape[1]), 4)
            per_q = np.stack([sh[:, ix].mean(1) for ix in q], 1)
            print('Shapley per quarter of the horizon (rows T, S, C; columns 1st..4th quarter; sum of a column = v(all) there):\n', per_q)
        if 'refit_mse' in r:
            Exp_Main._print_game(r['refit_mse'], 'Same game measured with MSE (robustness to the metric)', 'MSE')
        if 'renorm' in r:
            Exp_Main._print_game(r['renorm'], 'Shapley, trained weights renormalised over each subset (check)')
        if 'oob_share' in r:
            print('\n--- Outside the horizon basis (span of U plus constants), per channel and window ---')
            print('share of the truth outside the basis : %.4f' % r['oob_share'])
            print('                         temporal  spectral  channel  fused')
            print('corr with the truth outside basis  :', r['oob_corr'].numpy())
            print('R^2 of the truth outside basis     :', r['oob_r2'].numpy())
            print('energy outside basis / truth energy:', r['oob_energy'].numpy())

    @staticmethod
    def _jsonable(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): Exp_Main._jsonable(u) for k, u in v.items()}
        if isinstance(v, (list, tuple)):
            return [Exp_Main._jsonable(u) for u in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    def _save_results(self, setting, folder_path, mse, mae, r, theory, cost=None, explain=None):
        info = {'setting': setting, 'data': getattr(self.args, 'data_path', getattr(self.args, 'data', '')),
                'seq_len': self.args.seq_len, 'pred_len': self.args.pred_len,
                'seed': getattr(self.args, 'seed', None), 'mse': float(mse), 'mae': float(mae),
                'params': sum(p.numel() for p in self._wrapper().parameters())}
        if r is not None:
            info['diagnostics'] = r
        if theory:
            info['theory'] = theory
        if cost:
            info['inference_cost'] = cost
        if explain:
            info['explain'] = explain
        with open(os.path.join(folder_path, 'view_summary.json'), 'w') as f:
            json.dump(self._jsonable(info), f, indent=1)
        print('[SAVED]', os.path.join(folder_path, 'view_summary.json'))
    # ------------------------------------------------------------------------------------------
    # Checks of the four propositions (evaluation only)
    # ------------------------------------------------------------------------------------------
    def _param_breakdown(self):
        """Prop. 4: parameters per component at H and at 2H -> which components grow with the horizon."""
        def count(m):
            net = m.net
            parts = {'temporal': net.temporal, 'spectral': net.spectral, 'channel': net.channel, 'basis': net.basis}
            out = {k: sum(p.numel() for p in v.parameters()) for k, v in parts.items()}
            out['fusion'] = sum(p.numel() for n, p in net.named_parameters()
                                if n.startswith('fusion_') or n.startswith('amp_'))
            out['revin'] = sum(p.numel() for p in m.revin_layer.parameters())
            out['other'] = sum(p.numel() for p in m.parameters()) - sum(out.values())
            return out
        here = count(self._wrapper())
        try:
            args2 = copy.copy(self.args)
            args2.pred_len = 2 * self.args.pred_len
            twice = count(LGT.Model(args2))
        except Exception as ex:                     # e.g. a component that needs a GPU to build
            print('[THEORY] could not build the 2H model for the comparison:', ex)
            twice = None
        return here, twice

    @torch.no_grad()
    def _theory_checks(self, test_loader, folder_path):
        wrapper = self._wrapper()
        net = getattr(wrapper, 'net', None)
        if net is None or not hasattr(net, 'channel'):
            return {}
        ch = net.channel
        n_batches = int(getattr(self.args, 'theory_batches', 4))
        g = torch.Generator().manual_seed(2024)
        res = {'p1_err': 0.0, 'p1_row_min': 1.0, 'p1_row_max': 0.0, 'p1_diag': 0.0, 'p2_perm': 0.0}
        dev = {}                                         # subset size -> list of relative deviations
        U = None
        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= n_batches:
                break
            x = batch_x.float().to(self.device)
            if wrapper.revin:
                x = wrapper.revin_layer(x, 'norm')
            x1, _ = wrapper._inputs(x)
            xc = x1.permute(0, 2, 1).contiguous()        # [B, C, L] as seen by the channel view
            B, C, L = xc.shape
            if U is None:
                ym = wrapper._future_marks(batch_y_mark.float().to(self.device))
                U = net.basis(None if ym is None else ym[..., :2])[0].float().cpu()   # [H, R]
            if C < 2:
                continue

            # Prop. 1: r = A v exactly, A >= 0, zero diagonal, row sums in [0, 1)
            q, A = ch.relations(xc)
            res['p1_err'] = max(res['p1_err'], ch._last_check)
            rs = A.sum(-1)
            res['p1_row_min'] = min(res['p1_row_min'], float(rs.min()))
            res['p1_row_max'] = max(res['p1_row_max'], float(rs.max()))
            res['p1_diag'] = max(res['p1_diag'], float(A.diagonal(dim1=1, dim2=2).abs().max()))

            # Prop. 2 (i): permutation equivariance of the cross coefficients
            perm = torch.randperm(C, generator=g).to(self.device)
            _, cross = ch(xc)
            _, cross_p = ch(xc[:, perm])
            scale = cross.abs().max().clamp_min(1e-12)
            res['p2_perm'] = max(res['p2_perm'], float((cross_p - cross[:, perm]).abs().max() / scale))

            # Prop. 2 (ii): consistency in C, deviation of r_c on random channel subsets from r_c on all channels
            def ctx(xs):
                e = ch.embed(xs)
                qq, v, S, Z, T = ch._roles(e)
                return ch._cross(qq, v, S, Z, T)[0]
            r_full = ctx(xc)
            norm = r_full.norm(dim=-1).mean().clamp_min(1e-12)
            sizes = sorted({n for n in (C // 8, C // 4, C // 2) if n >= 4})
            for n in sizes:
                for _ in range(3):
                    idx = torch.randperm(C, generator=g)[:n].to(self.device)
                    d = (ctx(xc[:, idx]) - r_full[:, idx]).norm(dim=-1).mean() / norm
                    dev.setdefault(n, []).append(float(d))

        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        print('\n--- Theory checks (normalised inputs, %d test batches) ---' % n_batches)
        if dev or res['p1_row_max'] > 0:
            print('[Prop 1] max |A v - r| = %.2e | row sums of A in [%.4f, %.4f] | max |diag A| = %.1e'
                  % (res['p1_err'], res['p1_row_min'], res['p1_row_max'], res['p1_diag']))
            print('[Prop 2] permutation: max relative change of cross coefficients = %.2e' % res['p2_perm'])
        else:
            print('[Prop 1-2] skipped: one channel only')
        if len(dev) >= 2:
            ns = np.array(sorted(dev))
            ds = np.array([np.mean(dev[n]) for n in ns])
            slope = float(np.polyfit(np.log(ns), np.log(np.maximum(ds, 1e-12)), 1)[0])
            print('[Prop 2] subset size n :', ns, '| relative deviation of r_c :', ds,
                  '| log-log slope %.3f (theory -0.5)' % slope)
            res['p2_sizes'], res['p2_dev'], res['p2_slope'] = ns, ds, slope
        elif dev:
            print('[Prop 2] subset deviation:', {k: float(np.mean(v)) for k, v in dev.items()},
                  '(too few channels for a slope)')

        here, twice = self._param_breakdown()
        print('[Prop 4] parameters per component at H=%d:' % self.args.pred_len, here)
        if twice is not None:
            grow = {k: twice[k] - here[k] for k in here if twice[k] != here[k]}
            print('[Prop 4] components that change at 2H:', grow if grow else 'none')
        res['p4_params'], res['p4_params_2H'] = here, twice
        if U is not None:
            s = torch.linalg.svdvals(U.double())
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            ks = [k for k in (4, 8, 16, 32) if k <= len(e)]
            print('[Prop 4] basis U [%d x %d]: energy in top k singular values' % tuple(U.shape),
                  {k: round(float(e[k - 1]), 4) for k in ks})
            res['p4_basis_energy'] = e.numpy()
        np.save(folder_path + 'theory_checks.npy', res, allow_pickle=True)
        return res


    @staticmethod
    def _print_prop3(r):
        """Prop. 3: efficiency of the exact Shapley values of the re-fitted game."""
        a = r.get('refit')
        if a is None:
            return
        resid = float(a['shapley'].double().sum() - a['value'][7].double())
        print('[Prop 3] sum Shapley - v(all) = %.2e (efficiency, re-fitted game)' % resid)

    # ------------------------------------------------------------------------------------------
    # Cost at inference and interpretable quantities (evaluation only)
    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _measure_inference(self, test_loader, warmup=5):
        """Forward pass only (data already on the device, no analysis), over the whole test split.
        Returns time per batch and per window, throughput and peak memory of inference."""
        self.model.eval()
        cuda = self.device.type == 'cuda' if hasattr(self.device, 'type') else torch.cuda.is_available()
        batches = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batches.append((batch_x.float(), batch_x_mark.float(), batch_y_mark.float()))
        if not batches:
            return {}
        x0, xm0, ym0 = [t.to(self.device) for t in batches[0]]
        for _ in range(warmup):
            self.model(x0, x_mark=xm0, y_mark=ym0)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device=self.device)
            base = torch.cuda.memory_allocated(device=self.device)
        else:
            rss0 = psutil.Process(os.getpid()).memory_info().rss
        total, windows, times = 0.0, 0, []
        for bx, bxm, bym in batches:
            bx, bxm, bym = bx.to(self.device), bxm.to(self.device), bym.to(self.device)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model(bx, x_mark=bxm, y_mark=bym)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            total += dt
            windows += bx.shape[0]
        out = {'batches': len(times), 'batch_size': int(batches[0][0].shape[0]), 'windows': windows,
               'ms_per_batch': 1e3 * total / len(times), 'ms_per_batch_median': 1e3 * float(np.median(times)),
               'ms_per_window': 1e3 * total / windows, 'windows_per_s': windows / max(total, 1e-12),
               'total_s': total, 'device': str(self.device)}
        if cuda:
            out['peak_mem_mb'] = torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2
            out['peak_mem_above_weights_mb'] = (torch.cuda.max_memory_allocated(device=self.device) - base) / 1024 ** 2
        else:
            out['rss_increase_mb'] = (psutil.Process(os.getpid()).memory_info().rss - rss0) / 1024 ** 2
        print('\n--- Inference cost (forward only, %s, batch %d, %d windows) ---' % (out['device'], out['batch_size'], windows))
        print('time per batch %.3f ms (median %.3f) | per window %.4f ms | %.1f windows/s | total %.3f s'
              % (out['ms_per_batch'], out['ms_per_batch_median'], out['ms_per_window'], out['windows_per_s'], total))
        if cuda:
            print('peak GPU memory %.2f MB (%.2f MB above the weights and inputs already allocated)'
                  % (out['peak_mem_mb'], out['peak_mem_above_weights_mb']))
        else:
            print('CPU: process memory increase %.2f MB' % out['rss_increase_mb'])
        return out

    def _save_explain(self, folder_path, r, acc):
        """Interpretable quantities of the trained model -> explain.npz and figures (pdf + png)."""
        net = getattr(self._wrapper(), 'net', None)
        if net is None or not hasattr(net, 'explain'):
            return {}
        info = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in net.explain().items()
                if k != 'channel_role_membership'}
        if acc.get('n_role', 0) > 0:
            info['channel_role_membership'] = (acc['role'] / acc['n_role']).numpy()      # [C, G] test mean
            p = info['channel_role_membership'].mean(0)
            info['channel_effective_roles'] = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        if acc.get('n_gain', 0) > 0:
            info['spectral_effective_gain'] = (acc['gain'] / acc['n_gain']).numpy()     # [F] gain x Wiener, test mean
            info['input_spectral_entropy_mean'] = float(acc['omega'] / acc['n_gain'])
        if r is not None:
            info['fusion_weight_horizon'] = r['weight_horizon'].numpy()                   # [3, H] realised weights
            if 'refit' in r:
                info['shapley_horizon'] = r['refit']['shapley_horizon'].numpy()           # [3, H]
        np.savez(os.path.join(folder_path, 'explain.npz'), **{k: np.asarray(v) for k, v in info.items()})

        names = ['temporal', 'spectral', 'channel']
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            def save(fig, name):
                fig.tight_layout()
                for ext in ('pdf', 'png'):
                    fig.savefig(os.path.join(folder_path, name + '.' + ext), dpi=200)
                plt.close(fig)

            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            W = info.get('fusion_weight_horizon', info['fusion_weights'])
            h = np.arange(1, W.shape[1] + 1)
            for v in range(3):
                ax[0].plot(h, W[v], label=names[v])
            ax[0].set_xlabel('horizon step'); ax[0].set_ylabel('mean fusion weight'); ax[0].legend()
            if 'shapley_horizon' in info:
                for v in range(3):
                    ax[1].plot(h, info['shapley_horizon'][v], label=names[v])
                ax[1].axhline(0, color='k', lw=0.5)
                ax[1].set_xlabel('horizon step'); ax[1].set_ylabel('Shapley value (MAE)')
            else:
                ax[1].plot(h, info['amplitude_horizon']); ax[1].set_xlabel('horizon step'); ax[1].set_ylabel('amplitude s(h)')
            save(fig, 'fig_fusion_horizon')

            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            f = np.arange(len(info['spectral_gain']))
            ax[0].plot(f, info['spectral_gain'], label='learned gain sigmoid(a)')
            if 'spectral_effective_gain' in info:
                ax[0].plot(f, info['spectral_effective_gain'], label='effective gain (x Wiener), test mean')
            ax[0].set_xlabel('frequency index'); ax[0].set_ylabel('gain'); ax[0].legend()
            ax[1].semilogy(f, info['spectral_noise_floor'], label='noise floor N(w)')
            ax[1].set_xlabel('frequency index'); ax[1].set_ylabel('noise floor'); ax[1].legend()
            save(fig, 'fig_spectral_filter')

            if 'channel_role_membership' in info:
                M = info['channel_role_membership']
                fig, ax = plt.subplots(figsize=(6, max(2.5, min(10, 0.12 * M.shape[0] + 1.5))))
                im = ax.imshow(M, aspect='auto', cmap='viridis')
                ax.set_xlabel('role'); ax.set_ylabel('channel'); fig.colorbar(im, ax=ax, label='membership q(g|c)')
                save(fig, 'fig_channel_roles')
        except Exception as ex:
            print('[EXPLAIN] figures skipped:', ex)

        print('\n--- Interpretable quantities (saved to explain.npz and fig_*.pdf/png) ---')
        print('channel: temperature %.4f | role mixing per layer %s | effective number of roles %s'
              % (info['channel_temperature'], np.round(info['channel_role_mixing'], 4),
                 ('%.2f' % info['channel_effective_roles']) if 'channel_effective_roles' in info else '-'))
        print('fusion: entropy coefficients (T, S, C) %s | amplitude entropy coefficient %.4f'
              % (np.round(info['fusion_entropy_coef'], 4), info['amplitude_entropy_coef']))
        if 'input_spectral_entropy_mean' in info:
            print('mean spectral entropy of the test inputs %.4f' % info['input_spectral_entropy_mean'])
        g = info.get('spectral_effective_gain', info['spectral_gain'])
        top = np.argsort(-g)[:5]
        print('spectral: frequencies kept most (index: gain)', {int(k): round(float(g[k]), 3) for k in top})
        return {k: v for k, v in info.items() if np.ndim(v) <= 1 or k == 'fusion_weight_horizon'}

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            path = os.path.join(ckpt_dir, 'model.pth')
            if not os.path.exists(path):
                path = os.path.join(ckpt_dir, 'checkpoint.pth')
            self._wrapper().load_state_dict(torch.load(path, map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # test_time = 0 # For computational cost analysis
        wrapper = self._wrapper()
        analyse = hasattr(wrapper, 'forward_views') and self.args.features != 'MS'
        view_diag = ViewDiagnostics() if analyse else None
        cost = self._measure_inference(test_loader) if getattr(self.args, 'measure_inference', 1) else {}
        coal_fusion = None
        if analyse and hasattr(wrapper, 'coalition_forecasts'):
            coal_fusion = self._refit_coalitions()          # views frozen, one fusion fitted per subset
        acc = {}                                            # test means of input-dependent interpretable quantities
        net = getattr(wrapper, 'net', None)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # temp = time.time() # For computational cost analysis
                # calendar of input / target timestamps only (no target values -> no leakage)
                result = self.model(batch_x, x_mark=batch_x_mark, y_mark=batch_y_mark)
                if isinstance(result, tuple):
                    outputs, _ = result
                else:
                    outputs = result
                # test_time += time.time() - temp # For computational cost analysis
                if net is not None and hasattr(net, 'spectral'):
                    # read right after the model's own forward (eval mode): role membership, Wiener gain, entropy
                    if getattr(net.channel, '_last_assign', None) is not None:
                        q = net.channel._last_assign
                        acc['role'] = acc.get('role', 0) + q.sum(0).double().cpu()
                        acc['n_role'] = acc.get('n_role', 0) + q.shape[0]
                    P = net.spectral._last_power                                            # [B, C, F]
                    T = net.spectral.norm.normalized_shape[0]
                    g = torch.sigmoid(net.spectral.amp_gate) * P / (P + F_nn.softplus(net.spectral.noise_rho) * T)
                    acc['gain'] = acc.get('gain', 0) + g.mean(1).sum(0).double().cpu()
                    acc['n_gain'] = acc.get('n_gain', 0) + P.shape[0]
                    acc['omega'] = acc.get('omega', 0) + float(net._last_omega.mean(1).sum())

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if view_diag is not None:
                    # evaluation-only analysis: targets are compared with each view, never trained on
                    vd = wrapper.forward_views(batch_x, x_mark=batch_x_mark, y_mark=batch_y_mark)
                    coal = None
                    if coal_fusion is not None:
                        coal = wrapper.coalition_forecasts(coal_fusion, batch_x, y_mark=batch_y_mark)
                    U = wrapper.horizon_basis(batch_y_mark) if hasattr(wrapper, 'horizon_basis') else None
                    view_diag.update(vd['views'], vd['weights'], batch_y, x=batch_x, cross=vd['cross'],
                                     coal=coal, U=U)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        print("\n--- Model Statistics ---")
        test_model_size(self.model)
        test_params_memory(self.model, (batch_x.shape[1], batch_x.shape[2]))
        preds = np.array(preds)
        trues = np.array(trues)
        # preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        # trues = np.concatenate(trues, axis=0) # without the "drop-last" trick

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        r = None
        if view_diag is not None and view_diag.n > 0:
            r = view_diag.result()
            self._print_diagnostics(r)
            self._print_prop3(r)
        explain = self._save_explain(folder_path, r, acc)
        theory = {}
        if getattr(self.args, 'theory_batches', 4) > 0:
            theory = self._theory_checks(test_loader, folder_path)
        self._save_results(setting, folder_path, mse, mae, r, theory, cost, explain)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return