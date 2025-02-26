import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

config_file_path = './config/config.yaml'

with open(config_file_path, 'r') as config_file:
    config_data = yaml.safe_load(config_file)

def get_pnl(df_target_pos, df_match_price, fee_ratio):
    holding_arr = df_target_pos.values
    match_price_arr = df_match_price.values
    profit_without_fee_arr = np.full(holding_arr.shape, 0.)
    holding_delta_arr = np.full(holding_arr.shape, 0.)
    fee_arr = np.full(holding_arr.shape, 0.)
    profit_without_fee_arr[1:, :] = holding_arr[:-1, :] * (match_price_arr[1:, :] - match_price_arr[:-1, :])
    holding_delta_arr[1:, :] = np.abs(holding_arr[1:, :] - holding_arr[:-1, :])
    fee_arr[1:, :] = holding_delta_arr[1:, :] * match_price_arr[1:, :] * fee_ratio
    profit_with_fee_arr = profit_without_fee_arr - fee_arr
    df_profit_without_fee_arr = pd.DataFrame(profit_without_fee_arr.cumsum(axis=0), columns=df_target_pos.columns, index=df_target_pos.index)
    df_profit_with_fee_arr = pd.DataFrame(profit_with_fee_arr.cumsum(axis=0), columns=df_target_pos.columns, index=df_target_pos.index)
    return df_profit_without_fee_arr, df_profit_with_fee_arr

def get_metrics(nv_arr):
    year_difference = len(nv_arr) / (12 * 365)
    nv_max_arr = np.maximum.accumulate(nv_arr)
    vol_year = np.std(nv_arr[1:] / nv_arr[:-1] - 1, ddof=1) * np.sqrt(365 * 12)
    drawdown_arr = nv_max_arr - nv_arr
    max_drawdown = np.max(drawdown_arr)
    AnnualRateOfRtn = (nv_arr[-1] - 1) / year_difference
    calmar_ratio = AnnualRateOfRtn / max_drawdown
    sharpe_ratio = AnnualRateOfRtn / vol_year
    return AnnualRateOfRtn, max_drawdown, calmar_ratio, sharpe_ratio

def get_performance(strategy_name, cash_init, df_profit, df_profit_without_fee, save_df = False, save_pic = False):
    df_profit = df_profit.sum(axis = 1)
    df_profit_15min = df_profit.resample("2H").last().ffill()

    df_profit_without_fee = df_profit_without_fee.sum(axis=1)
    df_profit_without_fee = df_profit_without_fee.resample("2H").last().ffill()
    time_arr = df_profit_15min.index.values

    nv_arr = df_profit_15min.values / cash_init + 1
    AnnualRateOfRtn, max_drawdown, calmar_ratio, sharpe_ratio = get_metrics(nv_arr)
    nv_without_fee_arr = df_profit_without_fee.values / cash_init + 1
    AnnualRateOfRtn_without_fee, max_drawdown_without_fee, calmar_ratio_without_fee, sharpe_ratio_without_fee = get_metrics(nv_without_fee_arr)
    if save_pic == True:
        fig = plt.figure(figsize=(20, 15))
        plt.plot(time_arr, nv_without_fee_arr)
        plt.plot(time_arr, nv_arr)
        plt.grid()
        plt.legend([f"AnnualRateOfRtn_without_fee: {AnnualRateOfRtn_without_fee:.2%}, max_drawdown_without_fee: {max_drawdown_without_fee:.2%}, calmar_ratio_without_fee: {calmar_ratio_without_fee}, sharpe_ratio_without_fee: {sharpe_ratio_without_fee}",
                    f"AnnualRateOfRtn: {AnnualRateOfRtn:.2%}, max_drawdown: {max_drawdown:.2%}, calmar_ratio: {calmar_ratio}, sharpe_ratio: {sharpe_ratio}",
                    ],
                   loc = "upper left")
        plt.suptitle(strategy_name)
        plt.savefig(f"{config_data['backtest_results_path']}/pngs/{strategy_name}.png")
        plt.close()
    if save_df == True:
        df_summary = pd.DataFrame(columns=["time", "nv_withfee"])
        df_summary['time'] = time_arr
        df_summary['nv_withfee'] = nv_arr
        df_summary['nv_withoutfee'] = nv_without_fee_arr
        df_summary.to_hdf(f"{config_data['backtest_results_path']}/h5s/{strategy_name}.h5", key="data")
    return time_arr, nv_arr, nv_without_fee_arr, AnnualRateOfRtn, max_drawdown, calmar_ratio, sharpe_ratio
