import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    return mae, mse
    # return mae, mse, rmse, mape, mspe, rse, corr

# def CORR(pred, true):
#     # true, pred: [N, pred_len, C]
#     # Tính per-timestep, avg over channels
    
#     true_mean = true.mean(0, keepdims=True)   # [1, pred_len, C]
#     pred_mean = pred.mean(0, keepdims=True)

#     a = true - true_mean  # [N, pred_len, C]
#     b = pred - pred_mean

#     # Pearson đúng
#     num = (a * b).sum(0)                           # [pred_len, C]
#     den = np.sqrt((a**2).sum(0)) * np.sqrt((b**2).sum(0))  # [pred_len, C]
#     den += 1e-12

#     corr_per_step = (num / den).mean(-1)           # [pred_len]  ∈ [-1, 1]
#     return corr_per_step